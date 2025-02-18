from typing import Any, List, Dict, Optional, Tuple, Type, NamedTuple, Pattern, Union
import re, json
from types import TracebackType

from llama_index.core.graph_stores.prompts import DEFAULT_CYPHER_TEMPALTE
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.graph_stores.utils import (
    clean_string_values,
    value_sanitize,
    LIST_LIMIT,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores.types import VectorStoreQuery
import psycopg2
import psycopg2.extras

class AgensQueryException(Exception):
    """Exception for the Agensgraph queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details

def execute_query(curs, query, error_message = "Error executing query"):
    try:
        curs.execute(query)
    except psycopg2.Error as e:
        raise AgensQueryException(
            {
                "message": error_message,
                "details": str(e),
            }
        )

def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns:
    dict: A new dictionary with all empty values removed.
    """
    # Create a new dictionary excluding empty values
    return {key: value for key, value in input_dict.items() if value}


BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
EXCLUDED_LABELS = ["ag_vertex"]
EXCLUDED_RELS = ["ag_edge"]
EXHAUSTIVE_SEARCH_LIMIT = 10000
# Threshold for returning all available prop values in graph schema
DISTINCT_VALUE_LIMIT = 10
CHUNK_SIZE = 1000
VECTOR_INDEX_NAME = "entity"
LONG_TEXT_THRESHOLD = 52

# Since we do not support multiple labels, we will maintain the extra labels as a list
# This function will be used in queries to append new labels to the existing list
# and ensure that the labels are unique
append_label_function = """
    CREATE OR REPLACE FUNCTION append_label(labels jsonb, new_label text) 
    RETURNS jsonb AS $$
    DECLARE
        updated_labels jsonb;
    BEGIN
        IF labels IS NULL OR jsonb_typeof(labels) <> 'array' THEN
            labels := '[]'::jsonb;
        END IF;

        updated_labels := (SELECT jsonb_agg(DISTINCT elem) 
                        FROM jsonb_array_elements_text(labels || to_jsonb(new_label)) elem);

        RETURN updated_labels;
    END;
    $$ LANGUAGE plpgsql;

"""
class AgensPropertyGraphStore(PropertyGraphStore):
    """
    AgensGraph Property Graph Store.

    This class implements a AgensGraph property graph store.
    """

    vertex_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\](\{.*\})")
    edge_regex: Pattern = re.compile(r"(\w+)\[(\d+\.\d+)\]\[(\d+\.\d+),\s*(\d+\.\d+)\](\{.*\})")


    def __init__(
        self,
        graph_name: str,
        conf: Dict[str, Any],
        sanitize_query_output: bool = True,
        enhanced_schema: bool = False,
        create_indexes: bool = True,
        create: bool = True
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name
        self.sanitize_query_output = sanitize_query_output
        self.enhanced_schema = enhanced_schema
        self.create_indexes = create_indexes
        self.connection = psycopg2.connect(**conf)

        with self._get_cursor() as curs:
            graph_id_query = (
                """SELECT oid as graphid FROM ag_graph WHERE graphname = '{}';""".format(
                    graph_name
                )
            )
            execute_query(curs, graph_id_query)
            graphid = curs.fetchone()

            if graphid is None:
                if create:
                    create_statement = """
                        CREATE GRAPH {};
                    """.format(graph_name)
                    execute_query(curs, create_statement, "Error creating graph")
                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                curs.execute(graph_id_query)
                data = curs.fetchone()

            self.graphid = data.graphid

            graph_path = """SET graph_path = '{}';""".format(self.graph_name)
            execute_query(curs, graph_path)

            execute_query(curs, append_label_function)

            # self.refresh_schema()
            # self.query(
            #     """
            #     CREATE VLABEL IF NOT EXISTS "%s";
            #     CREATE CONSTRAINT ON "%s" ASSERT n.id IS UNIQUE;
            #     """
            #     % (self.node_label, self.node_label)
            # )

            # self.verify_vector_support()
            # if create_indexes:
            #     self.structured_query(
            #         f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{BASE_NODE_LABEL}`)
            #         REQUIRE n.id IS UNIQUE;"""
            #     )
            #     self.structured_query(
            #         f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{BASE_ENTITY_LABEL}`)
            #         REQUIRE n.id IS UNIQUE;"""
            #     )

            #     if self._supports_vector_index:
            #         self.structured_query(
            #             f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            #             "FOR (m:__Entity__) ON m.embedding"
            #         )

    def _get_cursor(self) -> psycopg2.extras.NamedTupleCursor:
        cursor = self.connection.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        return cursor

    @property
    def client(self) -> Any:
        return self.connection

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes with matching values."""
        return []

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets with matching values."""
        return []

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        return []

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        return None
    
    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""
        return None

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""
        return [], []

    # TODO:
    # def refresh_schema()

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        # replace embedding value from None to empty list
        # and if other key, than to empty string
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                chunked_params = [remove_empty_values(row) for row in chunked_params]
                print(chunked_params)
                self.structured_query(
                    """
                    UNWIND {} AS row
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.id}})
                    SET c.text = row.text, c.labels = append_label(c.labels, 'Chunk')
                    WITH c, row
                    SET c += row.properties
                    WITH c, row.embedding AS embedding
                    WHERE embedding IS NOT NULL
                    SET c.embedding = embedding
                    RETURN count(*)
                    """.format(chunked_params, BASE_NODE_LABEL=BASE_NODE_LABEL),
                )

        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                chunked_params = [remove_empty_values(row) for row in chunked_params]
                print(chunked_params)
                # TODO: verify the query
                self.structured_query(
                    """
                    UNWIND {} AS row
                    MERGE (e:"{BASE_NODE_LABEL}" {{id: row.id}})
                    SET e += CASE WHEN row.properties IS NOT NULL THEN row.properties ELSE properties(e) END
                    SET e.name = CASE WHEN row.name IS NOT NULL THEN row.name ELSE e.name END,
                        e.labels = append_label(e.labels, '{BASE_ENTITY_LABEL}')
                    WITH e, row
                    SET e.labels = append_label(e.labels, row.label)
                    WITH e, row
                    WHERE row.embedding IS NOT NULL
                    SET e.embedding = row.embedding
                    WITH e, row
                    WHERE row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:"{BASE_NODE_LABEL}" {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:"MENTIONS"]-(c)
                    """.format(chunked_params,
                               BASE_NODE_LABEL=BASE_NODE_LABEL, 
                               BASE_ENTITY_LABEL=BASE_ENTITY_LABEL),
                )

    @staticmethod
    def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
        """
        Convert a record returned from an agensgraph query to a dictionary

        Args:
            record (): a record from an agensgraph query result

        Returns:
            Dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record._fields:
            v = getattr(record, k)

            # records comes back label[id]{properties} which must be parsed
            if isinstance(v, str):
                vertex = AgensPropertyGraphStore.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensPropertyGraphStore.vertex_regex.match(v)
                edge = AgensPropertyGraphStore.edge_regex.match(v)

                if vertex:
                    d[k] = json.loads(vertex.group(3))

                # convert edge from id-label->id by replacing id with node information
                # we only do this if the vertex was also returned in the query
                # this is an attempt to be consistent with neo4j implementation
                elif edge:
                    elabel, edge_id, start_id, end_id, properties = edge.groups()
                    d[k] = (
                        vertices.get(start_id, {}),
                        elabel,
                        vertices.get(end_id, {}),
                    )
                else:
                    try:
                        d[k] = json.loads(v)
                    except json.JSONDecodeError:
                        d[k] = v

            else:
                d[k] = v

        return d

    def structured_query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, executing it and
        converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query (not used in this implementation)

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """

        # execute the query, rolling back on an error
        with self._get_cursor() as curs:
            try:
                curs.execute(query)
                self.connection.commit()
            except psycopg2.Error as e:
                self.connection.rollback()
                raise AgensQueryException(
                    {
                        "message": "Error executing graph query: {}".format(query),
                        "detail": str(e),
                    }
                )
            try:
                data = curs.fetchall()
            except psycopg2.ProgrammingError:
                data = []  # Handle queries that don’t return data

            if data is None:
                result = []
            # convert to dictionaries
            else:
                result = [self._record_to_dict(d) for d in data]

            return result













































#     def validate_relationship_schema(self, relationship_schema: List[Triple]) -> None:
#         # Check that validation schema is a list of tuples as required by Kùzu for relationships
#         if not all(isinstance(item, tuple) for item in relationship_schema):
#             raise ValueError(
#                 "Please specify the relationship schema as "
#                 "a list of tuples, for example: [('PERSON', 'IS_CEO_OF', 'ORGANIZATION')]"
#             )

#     @property
#     def client(self) -> kuzu.Connection:
#         return self.connection

#     def get_entities(self) -> List[str]:
#         return sorted(
#             set(
#                 [rel[0] for rel in self.relationship_schema]
#                 + [rel[2] for rel in self.relationship_schema]
#             )
#         )

#     def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
#         entity_list: List[EntityNode] = []
#         chunk_list: List[ChunkNode] = []
#         node_tables = self.connection._get_node_table_names()

#         for item in nodes:
#             if isinstance(item, EntityNode):
#                 entity_list.append(item)
#             elif isinstance(item, ChunkNode):
#                 chunk_list.append(item)

#         for chunk in chunk_list:
#             upsert_chunk_node_query = """
#                 MERGE (c:Chunk {id: $id})
#                   SET c.text = $text,
#                       c.label = $label,
#                       c.embedding = $embedding,
#                       c.ref_doc_id = $ref_doc_id,
#                       c.creation_date = date($creation_date),
#                       c.last_modified_date = date($last_modified_date),
#                       c.file_name = $file_name,
#                       c.file_path = $file_path,
#                       c.file_size = $file_size,
#                       c.file_type = $file_type
#                 """

#             self.connection.execute(
#                 upsert_chunk_node_query,
#                 parameters={
#                     "id": chunk.id_,
#                     "text": chunk.text.strip(),
#                     "label": chunk.label,
#                     "embedding": chunk.embedding,
#                     "ref_doc_id": chunk.properties.get("ref_doc_id"),
#                     "creation_date": chunk.properties.get("creation_date"),
#                     "last_modified_date": chunk.properties.get("last_modified_date"),
#                     "file_name": chunk.properties.get("file_name"),
#                     "file_path": chunk.properties.get("file_path"),
#                     "file_size": chunk.properties.get("file_size"),
#                     "file_type": chunk.properties.get("file_type"),
#                 },
#             )

#         for entity in entity_list:
#             entity_label = entity.label if entity.label in node_tables else "Entity"
#             upsert_entity_node_query = f"""
#                 MERGE (e:{entity_label} {{id: $id}})
#                 SET e.label = $label,
#                     e.name = $name,
#                     e.embedding = $embedding,
#                     e.creation_date = date($creation_date),
#                     e.last_modified_date = date($last_modified_date),
#                     e.file_name = $file_name,
#                     e.file_path = $file_path,
#                     e.file_size = $file_size,
#                     e.file_type = $file_type,
#                     e.triplet_source_id = $triplet_source_id
#                 """

#             self.connection.execute(
#                 upsert_entity_node_query,
#                 parameters={
#                     "id": entity.name,
#                     "label": entity.label,
#                     "name": entity.name,
#                     "embedding": entity.embedding,
#                     "creation_date": entity.properties.get("creation_date"),
#                     "last_modified_date": entity.properties.get("last_modified_date"),
#                     "file_name": entity.properties.get("file_name"),
#                     "file_path": entity.properties.get("file_path"),
#                     "file_size": entity.properties.get("file_size"),
#                     "file_type": entity.properties.get("file_type"),
#                     "triplet_source_id": entity.properties.get("triplet_source_id"),
#                 },
#             )

#     def upsert_relations(self, relations: List[Relation]) -> None:
#         for rel in relations:
#             if self.has_structured_schema:
#                 src, rel_tbl_name, dst = utils.lookup_relation(
#                     rel.label, self.relationship_schema
#                 )
#             else:
#                 src, rel_tbl_name, dst = "Entity", "LINKS", "Entity"

#             # Connect entities to each other
#             self.connection.execute(
#                 f"""
#                 MATCH (a:{src} {{id: $source_id}}),
#                       (b:{dst} {{id: $target_id}})
#                 MERGE (a)-[r:{rel_tbl_name} {{label: $label}}]->(b)
#                     SET r.triplet_source_id = $triplet_source_id
#                 """,
#                 parameters={
#                     "source_id": rel.source_id,
#                     "target_id": rel.target_id,
#                     "triplet_source_id": rel.properties.get("triplet_source_id"),
#                     "label": rel.label,
#                 },
#             )
#             # Connect chunks to entities
#             self.connection.execute(
#                 f"""
#                 MATCH (a:{src} {{id: $source_id}}),
#                         (b:{dst} {{id: $target_id}}),
#                         (c:Chunk {{id: $triplet_source_id}})
#                 MERGE (c)-[:MENTIONS]->(a)
#                 MERGE (c)-[:MENTIONS]->(b)
#                 """,
#                 parameters={
#                     "source_id": rel.source_id,
#                     "target_id": rel.target_id,
#                     "triplet_source_id": rel.properties.get("triplet_source_id"),
#                 },
#             )

#     def structured_query(
#         self, query: str, param_map: Optional[Dict[str, Any]] = None
#     ) -> Any:
#         response = self.connection.execute(query, parameters=param_map)
#         column_names = response.get_column_names()
#         result = []
#         while response.has_next():
#             row = response.get_next()
#             result.append(dict(zip(column_names, row)))

#         if self.sanitize_query_output:
#             return value_sanitize(result)

#         return result

#     def vector_query(
#         self, query: VectorStoreQuery, **kwargs: Any
#     ) -> Tuple[List[LabelledNode], List[float]]:
#         raise NotImplementedError(
#             "Vector query is not currently implemented for KuzuPropertyGraphStore."
#         )

#     def get(
#         self,
#         properties: Optional[dict] = None,
#         ids: Optional[List[str]] = None,
#     ) -> List[LabelledNode]:
#         """Get nodes from the property graph store."""
#         cypher_statement = "MATCH (e) "

#         parameters = {}
#         if ids:
#             cypher_statement += "WHERE e.id in $ids "
#             parameters["ids"] = ids

#         return_statement = "RETURN e.*"
#         cypher_statement += return_statement
#         result = self.structured_query(cypher_statement, param_map=parameters)
#         result = result if result else []

#         nodes = []
#         for record in result:
#             # Text indicates a chunk node
#             # None on the label indicates an implicit node, likely a chunk node
#             if record.get("e.label") == "text_chunk":
#                 properties = {
#                     k: v for k, v in record.items() if k not in ["e.id", "e.text"]
#                 }
#                 text = record.get("e.text")
#                 nodes.append(
#                     ChunkNode(
#                         id_=record["e.id"],
#                         text=text,
#                         properties=utils.remove_empty_values(properties),
#                     )
#                 )
#             else:
#                 properties = {
#                     k: v for k, v in record.items() if k not in ["e.id", "e.name"]
#                 }
#                 name = record["e.name"] if record.get("e.name") else record["e.id"]
#                 label = record["e.label"] if record.get("e.label") else "Chunk"
#                 nodes.append(
#                     EntityNode(
#                         name=name,
#                         label=label,
#                         properties=utils.remove_empty_values(properties),
#                     )
#                 )
#         return nodes

#     def get_triplets(
#         self,
#         entity_names: Optional[List[str]] = None,
#         relation_names: Optional[List[str]] = None,
#         ids: Optional[List[str]] = None,
#     ) -> List[Triplet]:
#         # Construct the Cypher query
#         cypher_statement = "MATCH (e)-[r]->(t) "

#         params = {}
#         if entity_names or relation_names or ids:
#             cypher_statement += "WHERE "

#         if entity_names:
#             cypher_statement += "e.name in $entity_names "
#             params["entity_names"] = entity_names

#         if relation_names and entity_names:
#             cypher_statement += f"AND "
#         if relation_names:
#             cypher_statement += "r.label in $relation_names "
#             params[f"relation_names"] = relation_names

#         if ids:
#             cypher_statement += "e.id in $ids "
#             params["ids"] = ids

#         # Avoid returning a massive list of triplets that represent a large portion of the graph
#         # This uses the LIMIT constant defined at the top of the file
#         if not (entity_names or relation_names or ids):
#             return_statement = f"WHERE e.label <> 'text_chunk' RETURN * LIMIT {LIMIT};"
#         else:
#             return_statement = f"AND e.label <> 'text_chunk' RETURN * LIMIT {LIMIT};"

#         cypher_statement += return_statement

#         result = self.structured_query(cypher_statement, param_map=params)
#         result = result if result else []

#         triples = []
#         for record in result:
#             if record["e"]["_label"] == "Chunk":
#                 continue

#             src_table = record["e"]["_id"]["table"]
#             dst_table = record["t"]["_id"]["table"]
#             id_map = {src_table: record["e"]["id"], dst_table: record["t"]["id"]}
#             source = EntityNode(
#                 name=record["e"]["id"],
#                 label=record["e"]["_label"],
#                 properties=utils.get_filtered_props(record["e"], ["_id", "_label"]),
#             )
#             target = EntityNode(
#                 name=record["t"]["id"],
#                 label=record["t"]["_label"],
#                 properties=utils.get_filtered_props(record["t"], ["_id", "_label"]),
#             )
#             rel = Relation(
#                 source_id=id_map.get(record["r"]["_src"]["table"], "unknown"),
#                 target_id=id_map.get(record["r"]["_dst"]["table"], "unknown"),
#                 label=record["r"]["label"],
#             )
#             triples.append([source, rel, target])
#         return triples

#     def get_rel_map(
#         self,
#         graph_nodes: List[LabelledNode],
#         depth: int = 2,
#         limit: int = 30,
#         ignore_rels: Optional[List[str]] = None,
#     ) -> List[Triplet]:
#         triples = []

#         ids = [node.id for node in graph_nodes]
#         if len(ids) > 0:
#             # Run recursive query
#             response = self.structured_query(
#                 f"""
#                 MATCH (e)
#                 WHERE e.id IN $ids
#                 MATCH (e)-[rel*1..{depth} (r, n | WHERE r.label <> "MENTIONS") ]->(other)
#                 RETURN *
#                 LIMIT {limit};
#                 """,
#                 param_map={"ids": ids},
#             )
#         else:
#             response = self.structured_query(
#                 f"""
#                 MATCH (e)
#                 MATCH (e)-[rel*1..{depth} (r, n | WHERE r.label <> "MENTIONS") ]->(other)
#                 RETURN *
#                 LIMIT {limit};
#                 """
#             )

#         ignore_rels = ignore_rels or []
#         for record in response:
#             for item in record["rel"]["_rels"]:
#                 if item["label"] in ignore_rels:
#                     continue

#                 src_table = item["_src"]["table"]
#                 dst_table = item["_src"]["table"]
#                 id_map = {
#                     src_table: record["e"]["_id"],
#                     dst_table: record["other"]["id"],
#                 }
#                 source = EntityNode(
#                     name=record["e"]["name"],
#                     label=record["e"]["_label"],
#                     properties=utils.get_filtered_props(
#                         record["e"], ["_id", "name", "_label"]
#                     ),
#                 )
#                 target = EntityNode(
#                     name=record["other"]["name"],
#                     label=record["other"]["_label"],
#                     properties=utils.get_filtered_props(
#                         record["e"], ["_id", "name", "_label"]
#                     ),
#                 )
#                 rel = Relation(
#                     source_id=id_map.get(item["_src"]["table"], "unknown"),
#                     target_id=id_map.get(item["_dst"]["table"], "unknown"),
#                     label=item["label"],
#                 )
#                 triples.append([source, rel, target])

#         return triples

#     def delete(
#         self,
#         entity_names: Optional[List[str]] = None,
#         relation_names: Optional[List[str]] = None,
#         properties: Optional[dict] = None,
#         ids: Optional[List[str]] = None,
#     ) -> None:
#         """Delete nodes and relationships from the property graph store."""
#         if entity_names:
#             self.structured_query(
#                 "MATCH (n) WHERE n.name IN $entity_names DETACH DELETE n",
#                 param_map={"entity_names": entity_names},
#             )

#         if ids:
#             self.structured_query(
#                 "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
#                 param_map={"ids": ids},
#             )

#         if relation_names:
#             for rel in relation_names:
#                 src, _, dst = utils.lookup_relation(rel, self.relationship_schema)
#                 self.structured_query(
#                     f"""
#                     MATCH (:{src})-[r {{label: $label}}]->(:{dst})
#                     DELETE r
#                     """,
#                     param_map={"label": rel},
#                 )

#         if properties:
#             assert isinstance(
#                 properties, dict
#             ), "`properties` should be a key-value mapping."
#             cypher = "MATCH (e) WHERE "
#             prop_list = []
#             params = {}
#             for i, prop in enumerate(properties):
#                 prop_list.append(f"e.`{prop}` = $property_{i}")
#                 params[f"property_{i}"] = properties[prop]
#             cypher += " AND ".join(prop_list)
#             self.structured_query(cypher + " DETACH DELETE e", param_map=params)

#     def get_schema(self) -> Any:
#         """
#         Returns a structured schema of the property graph store.

#         The schema contains `node_props`, `rel_props`, and `relationships` keys and
#         the associated metadata.
#         Example output:
#         {
#             'node_props': {'Chunk': [{'property': 'id', 'type': 'STRING'},
#                                     {'property': 'text', 'type': 'STRING'},
#                                     {'property': 'label', 'type': 'STRING'},
#                                     {'property': 'embedding', 'type': 'DOUBLE'},
#                                     {'property': 'properties', 'type': 'STRING'},
#                                     {'property': 'ref_doc_id', 'type': 'STRING'}],
#                             'Entity': [{'property': 'id', 'type': 'STRING'},
#                                     {'property': 'name', 'type': 'STRING'},
#                                     {'property': 'label', 'type': 'STRING'},
#                                     {'property': 'embedding', 'type': 'DOUBLE'},
#                                     {'property': 'properties', 'type': 'STRING'}]},
#             'rel_props': {'SOURCE': [{'property': 'label', 'type': 'STRING'}]},
#             'relationships': [{'end': 'Chunk', 'start': 'Chunk', 'type': 'SOURCE'}]
#         }
#         """
#         current_table_schema = {"node_props": {}, "rel_props": {}, "relationships": []}
#         node_tables = self.connection._get_node_table_names()
#         for table_name in node_tables:
#             node_props = self.connection._get_node_property_names(table_name)
#             current_table_schema["node_props"][table_name] = []
#             for prop, attr in node_props.items():
#                 schema = {}
#                 schema["property"] = prop
#                 schema["type"] = attr["type"]
#                 current_table_schema["node_props"][table_name].append(schema)

#         rel_tables = self.connection._get_rel_table_names()
#         for i, table in enumerate(rel_tables):
#             table_name = table["name"]
#             prop_values = self.connection.execute(
#                 f"MATCH ()-[r:{table_name}]->() RETURN distinct r.label AS label;"
#             )
#             while prop_values.has_next():
#                 rel_label = prop_values.get_next()[0]
#                 src, dst = rel_tables[i]["src"], rel_tables[i]["dst"]
#                 current_table_schema["relationships"].append(
#                     {"start": src, "type": rel_label, "end": dst}
#                 )
#                 current_table_schema["rel_props"][rel_label] = []
#                 table_details = self.connection.execute(
#                     f"CALL TABLE_INFO('{table_name}') RETURN *;"
#                 )
#                 while table_details.has_next():
#                     props = table_details.get_next()
#                     rel_props = {}
#                     rel_props["property"] = props[1]
#                     rel_props["type"] = props[2]
#                     current_table_schema["rel_props"][rel_label].append(rel_props)

#         self.structured_schema = current_table_schema

#         return self.structured_schema

#     def get_schema_str(self) -> str:
#         schema = self.get_schema()

#         formatted_node_props = []
#         formatted_rel_props = []

#         # Format node properties
#         for label, props in schema["node_props"].items():
#             props_str = ", ".join(
#                 [f"{prop['property']}: {prop['type']}" for prop in props]
#             )
#             formatted_node_props.append(f"{label} {{{props_str}}}")

#         # Format relationship properties
#         for type, props in schema["rel_props"].items():
#             props_str = ", ".join(
#                 [f"{prop['property']}: {prop['type']}" for prop in props]
#             )
#             formatted_rel_props.append(f"{type} {{{props_str}}}")

#         # Format relationships
#         formatted_rels = [
#             f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})"
#             for rel in schema["relationships"]
#         ]

#         return "\n".join(
#             [
#                 "Node properties:",
#                 "\n".join(formatted_node_props),
#                 "Relationship properties:",
#                 "\n".join(formatted_rel_props),
#                 "The relationships:",
#                 "\n".join(formatted_rels),
#             ]
#         )


# KuzuPGStore = KuzuPropertyGraphStore
