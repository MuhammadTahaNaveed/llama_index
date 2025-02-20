from typing import Any, List, Dict, Optional, Tuple, Type, NamedTuple, Pattern, Union
import re, json
from types import TracebackType
import logging

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

def remove_nones(input_dict):
    """
    Remove entries with None values from the dictionary.

    Parameters:
    input_dict (dict): The dictionary from which None values need to be removed.

    Returns:
    dict: A new dictionary with all None values removed.
    """
    return {key: value for key, value in input_dict.items() if value is not None}


BASE_ENTITY_LABEL = "__Entity__"
BASE_NODE_LABEL = "__Node__"
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
    BEGIN
        IF labels IS NULL OR jsonb_typeof(labels) <> 'array' THEN
            labels := '[]'::jsonb;
        END IF;

        IF NOT labels @> to_jsonb(new_label) THEN
            RETURN labels || jsonb_build_array(new_label);
        ELSE
            RETURN labels;
        END IF;
    END;
    $$ LANGUAGE plpgsql;

"""

label_catalog = """
CREATE TABLE IF NOT EXISTS label_catalog (
    graph_id oid PRIMARY KEY,
    labels jsonb DEFAULT '[]'::jsonb
);

"""

track_labels = """
CREATE OR REPLACE FUNCTION track_labels()
RETURNS TRIGGER AS $$
DECLARE
    graphid OID := {}::oid;
    new_labels JSONB;
BEGIN
    INSERT INTO label_catalog (graph_id, labels)
    VALUES (graphid, '[]'::jsonb)
    ON CONFLICT (graph_id) DO NOTHING;

    IF NEW.properties ? 'labels' THEN
        new_labels := NEW.properties->'labels';
        new_labels := (
            SELECT jsonb_agg(elems)
            FROM jsonb_array_elements_text(new_labels) AS elems
            WHERE elems NOT IN ('__Node__', '__Entity__')
        );
    ELSE
        new_labels := '[]'::jsonb;
    END IF;

    UPDATE label_catalog
    SET labels = (
        SELECT jsonb_agg(DISTINCT elems)
        FROM jsonb_array_elements(COALESCE(labels, '[]'::jsonb) || COALESCE(new_labels, '[]'::jsonb)) AS elems
    )
    WHERE graph_id = graphid;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

"""

track_labels_trigger = """
CREATE TRIGGER track_labels_trigger
AFTER INSERT OR UPDATE ON "{}"."__Node__"
FOR EACH ROW EXECUTE FUNCTION track_labels();

"""

node_properties_query = f"""
    MATCH (a:"{BASE_NODE_LABEL}")
    UNWIND a.labels AS label
    UNWIND keys(properties(a)) AS prop
    WITH label, prop, properties(a)[prop] AS value 
    WHERE prop != 'labels'
    WITH                
        label,
        prop AS property,
        COLLECT(DISTINCT value) AS values,
        COUNT(DISTINCT value) AS distinct_count
    WHERE label != '{BASE_ENTITY_LABEL}' 
    RETURN label, COLLECT({{'property': property, 'values':values, 'distinct_count': distinct_count}}) as props;
"""

rel_query = f"""
    MATCH (start_node)-[r]->(end_node)
    WITH DISTINCT start_node.labels AS start_labels, type(r) AS relationship_type, end_node.labels AS end_labels
    UNWIND start_labels AS start_label
    UNWIND end_labels AS end_label
    WITH DISTINCT start_label, relationship_type, end_label
    WHERE start_label != '{BASE_ENTITY_LABEL}' AND end_label != '{BASE_ENTITY_LABEL}'
    RETURN {{start: start_label, type: relationship_type, end: end_label}} AS output
"""

logger = logging.getLogger(__name__)

class AgensPropertyGraphStore(PropertyGraphStore):
    """
    AgensGraph Property Graph Store.

    This class implements a AgensGraph property graph store.
    """

    types = {
        "str": "STRING",
        "float": "DOUBLE",
        "int": "INTEGER",
        "list": "LIST",
        "dict": "MAP",
        "bool": "BOOLEAN",
    }

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
            data = curs.fetchone()

            if data is None:
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

            # Create functions, triggers and catalog to handle multiple labels
            execute_query(curs, append_label_function)
            execute_query(curs, label_catalog)
            execute_query(curs, track_labels.format(self.graphid))
            execute_query(curs, f'CREATE VLABEL IF NOT EXISTS "{BASE_NODE_LABEL}"')
            execute_query(curs, track_labels_trigger.format(self.graph_name))
            self.connection.commit()
            self.refresh_schema()

            # self.verify_vector_support()
            if create_indexes:
                self.structured_query(
                    f"""CREATE CONSTRAINT ON "{BASE_NODE_LABEL}"
                        ASSERT n.id IS UNIQUE;"""
                )
            #     self.structured_query(
            #         f"""CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{BASE_ENTITY_LABEL}`)
            #         REQUIRE n.id IS UNIQUE;"""
            #     )

            #     if self._supports_vector_index:
            #         self.structured_query(
            #             f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            #             "FOR (m:__Entity__) ON m.embedding"
            #         )
            # Also add constraint to ensure that labels property is always a jsonb array

    def _get_cursor(self) -> psycopg2.extras.NamedTupleCursor:
        cursor = self.connection.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        return cursor

    @property
    def client(self) -> Any:
        return self.connection

    def refresh_schema(self) -> None:
        """
        Refresh the graph schema information by updating the available
        labels, relationships, and properties
        """

        # fetch graph schema information
        n_labels, e_labels = self._get_labels()

        node_properties = self._get_node_properties()
        edge_properties = self._get_edge_properties(e_labels)
        triple_schema = self._get_triples()

        # update the dictionary representation
        self.structured_schema = {
            "node_props": node_properties,
            "rel_props": {el["type"]: el["properties"] for el in edge_properties},
            "relationships": triple_schema,
            "metadata": {},
        }

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the FalkorDBGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
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
                chunked_params = [remove_nones(row) for row in chunked_params]
                self.structured_query(
                    """
                    UNWIND {} AS row
                    MERGE (c:"{BASE_NODE_LABEL}" {{id: row.id}})
                    SET c.text = row.text, c.labels = append_label(c.labels, 'Chunk')
                    WITH c, row
                    SET c += row.properties, c.embedding = row.embedding
                    RETURN count(*)
                    """.format(chunked_params, BASE_NODE_LABEL=BASE_NODE_LABEL),
                )

        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                chunked_params = [remove_nones(row) for row in chunked_params]
                self.structured_query(
                    """
                    UNWIND {chunked_params} AS row
                    MERGE (e:"{BASE_NODE_LABEL}" {{id: row.id}})
                    SET e += CASE WHEN row.properties IS NOT NULL THEN row.properties ELSE properties(e) END
                    SET e.name = CASE WHEN row.name IS NOT NULL THEN row.name ELSE e.name END,
                        e.labels = append_label(e.labels, '{BASE_ENTITY_LABEL}')
                    WITH e, row
                    SET e.labels = append_label(e.labels, row.label);

                    UNWIND {chunked_params} AS row
                    MATCH (e:"{BASE_NODE_LABEL}" {{id: row.id}})
                    WHERE row.embedding IS NOT NULL
                    SET e.embedding = row.embedding
                    WITH e, row
                    WHERE row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:"{BASE_NODE_LABEL}" {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:"MENTIONS"]-(c)
                    """.format(chunked_params=chunked_params,
                               BASE_NODE_LABEL=BASE_NODE_LABEL, 
                               BASE_ENTITY_LABEL=BASE_ENTITY_LABEL),
                )

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        print("Upserting relations")
        params = [r.dict() for r in relations]
        for index in range(0, len(params), CHUNK_SIZE):
            chunked_params = params[index : index + CHUNK_SIZE]
            for param in chunked_params:
                formatted_properties = ", ".join(
                    [f"{key}: {value!r}" for key, value in param["properties"].items()]
                )
                self.structured_query(
                    f"""
                    MERGE (source: "{BASE_NODE_LABEL}" {{id: '{param["source_id"]}'}})
                    ON CREATE SET source.labels = append_label(source.labels, 'Chunk')
                    MERGE (target: "{BASE_NODE_LABEL}" {{id: '{param["target_id"]}'}})
                    ON CREATE SET target.labels = append_label(target.labels, 'Chunk')
                    WITH source, target
                    MERGE (source)-[r:"{param["label"]}"]->(target)
                    SET r += {{{formatted_properties}}}
                    RETURN count(*)
                    """
                )

    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes."""
        wrapper = """SELECT t.name, t.type, t.properties - 'labels' AS properties FROM ({})t"""
        cypher_statement = f'MATCH (e:"{BASE_NODE_LABEL}") '

        cypher_statement += "WHERE e.id IS NOT NULL "

        if ids:
            cypher_statement += "AND e.id IN {} ".format(ids)

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                property = properties[prop] if not isinstance(properties[prop], str) else f"'{properties[prop]}'"
                prop_list.append(f'e."{prop}" = {property}')
            cypher_statement += " AND " + " AND ".join(prop_list)

        return_statement = f"""
            WITH e, e.labels as labels
            RETURN
            e.id AS name,
            CASE
                WHEN '{BASE_ENTITY_LABEL}' IN labels THEN
                    CASE
                        WHEN length(labels) > 2 THEN labels[2]
                        WHEN length(labels) > 1 THEN labels[1]
                        ELSE NULL
                    END
                ELSE labels[0]
            END AS type,
            properties(e) AS properties
        """
        cypher_statement += return_statement
        response = self.structured_query(wrapper.format(cypher_statement))
        response = response if response else []

        nodes = []
        for record in response:
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=remove_empty_values(record["properties"]),
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=remove_empty_values(record["properties"]),
                    )
                )

        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        cypher_statement = "MATCH (e)-[r]->(t) "
        cypher_statement += f"WHERE '{BASE_ENTITY_LABEL}' IN e.labels "

        if entity_names or relation_names or properties or ids:
            cypher_statement += "AND "

        if entity_names:
            cypher_statement += "e.name IN {} ".format(entity_names)

        if relation_names and entity_names:
            cypher_statement += "AND "

        if relation_names:
            cypher_statement += "type(r) IN {} ".format(relation_names)

        if ids:
            cypher_statement += "e.id IN {} ".format(ids)

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                property = properties[prop] if not isinstance(properties[prop], str) else f"'{properties[prop]}'"
                prop_list.append(f'e."{prop}" = {property}')
            cypher_statement += " AND ".join(prop_list)

        return_statement = f"""
        AND NOT ANY(label IN e.labels WHERE label = 'Chunk')
            WITH *, e.labels as e_labels, t.labels as t_labels
            RETURN type(r) as type, properties(r) as rel_prop, e.id as source_id,
            CASE
                WHEN '{BASE_ENTITY_LABEL}' IN e_labels THEN
                    CASE
                        WHEN length(e_labels) > 2 THEN e_labels[2]
                        WHEN length(e_labels) > 1 THEN e_labels[1]
                        ELSE NULL
                    END
                ELSE e_labels[0]
            END AS source_type,
            properties(e) AS source_properties,
            t.id as target_id,
            CASE
                WHEN '{BASE_ENTITY_LABEL}' IN t_labels THEN
                    CASE
                        WHEN length(t_labels) > 2 THEN t_labels[2]
                        WHEN length(t_labels) > 1 THEN t_labels[1]
                        ELSE NULL
                    END
                ELSE t_labels[0]
            END AS target_type, properties(t) AS target_properties LIMIT 100
        """

        cypher_statement += return_statement
        wrapper = """
                    SELECT t.type,
                           t.rel_prop,
                           t.source_id,
                           t.source_type,
                           t.source_properties - 'labels' AS source_properties,
                           t.target_id,
                           t.target_type,
                           t.target_properties - 'labels' AS target_properties
                    FROM ({})t;
        """
        data = self.structured_query(wrapper.format(cypher_statement))
        data = data if data else []

        triplets = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_prop"]),
            )
            triplets.append([source, rel, target])
        return triplets

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        triples = []

        ids = [node.id for node in graph_nodes]
        cypher_statement = f"""
            UNWIND {[0] if len(ids) == 1 else f'range(0, {len(ids)} - 1)::jsonb'} AS idx
            MATCH (e:"{BASE_NODE_LABEL}")
            WHERE e.id = {ids}[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel, idx, collect(type(rel)) AS types
            WHERE all(x IN types WHERE x <> 'MENTIONS')
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel AS rel_properties,
                endNode(rel) AS endNode,
                idx,
                startNode(rel).labels AS source_labels,
                endNode(rel).labels AS target_labels
            LIMIT {limit}
            RETURN source.id AS source_id,
                CASE
                    WHEN '{BASE_ENTITY_LABEL}' IN source_labels THEN
                        CASE
                            WHEN length(source_labels) > 2 THEN source_labels[2]
                            WHEN length(source_labels) > 1 THEN source_labels[1]
                            ELSE NULL
                        END
                    ELSE source_labels[0]
                END AS source_type,
                properties(source) AS source_properties,
                type,
                properties(rel_properties) as rel_properties,
                endNode.id AS target_id,
                CASE
                    WHEN '{BASE_ENTITY_LABEL}' IN target_labels THEN
                        CASE
                            WHEN length(target_labels) > 2 THEN target_labels[2]
                            WHEN length(target_labels) > 1 THEN target_labels[1] ELSE NULL
                        END
                    ELSE target_labels[0]
                END AS target_type,
                properties(endNode) AS target_properties,
                idx
            ORDER BY idx
            LIMIT {limit}
            """
        wrapper = """SELECT t.source_id,
                            t.source_type,
                            t.source_properties - 'labels' AS source_properties,
                            t.type,
                            t.rel_properties,
                            t.target_id,
                            t.target_type,
                            t.target_properties - 'labels' AS target_properties
                      FROM ({})t;
          """
        response = self.structured_query(wrapper.format(cypher_statement))
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_properties"]),
            )
            triples.append([source, rel, target])

        return triples
    
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
                print(query)
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

    def _get_node_properties(self) -> List[Dict[str, Any]]:
        node_properties = {}
        with self._get_cursor() as curs:
            execute_query(curs, node_properties_query)
            rows = curs.fetchall()

            for row in rows:
                props = row.props
                for prop in props:
                    prop["type"] = self.types[type(prop["values"][0]).__name__]
                
                node_properties[row.label] = props

        return node_properties

    def _get_edge_properties(self, e_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch a list of available edge properties by edge label to be used
        as context for an llm

        Args:
            e_labels (List[str]): a list of edge labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of edge labels
                and their corresponding properties in the form
                "{
                    'labels': <edge_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """
        # cypher query to fetch properties of a given label
        edge_properties_query = """
            MATCH ()-[e:"{e_label}"]->()
            RETURN properties(e) AS props
            LIMIT 100
        """
        edge_properties = []
        with self._get_cursor() as curs:
            for label in e_labels:
                q = edge_properties_query.format(
                    e_label=label
                )

                execute_query(curs, q)
                data = curs.fetchall()

                # build a set of distinct properties
                s = set({})
                for d in data:
                    for k, v in d.props.items():
                        s.add((k, self.types[type(v).__name__]))

                np = {
                    "properties": [{"property": k, "type": v} for k, v in s],
                    "type": label,
                }
                edge_properties.append(np)

        return edge_properties

    def _get_triples(self) -> List[Dict[str, str]]:
        """
        Get a set of distinct relationship types (as a list of dicts) in the graph
        to be used as context by an llm.

        Returns:
            List[Dict[str, str]]: relationships as a list of dicts in the format
                "{'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}"
        """

        triple_schema = []
        triple_schema = self.structured_query(rel_query)
        if len(triple_schema) == 0:
            return []
        
        triple_schema = [item["output"] for item in triple_schema]
        return triple_schema

    def _get_triples_str(self) -> List[str]:
        """
        Get a set of distinct relationship types (as a list of strings) in the graph
        to be used as context by an llm.

        Returns:
            List[str]: relationships as a list of strings in the format
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """

        triples = self._get_triples()
        return self._format_triples(triples)

    @staticmethod
    def _format_triples(triples: List[Dict[str, str]]) -> List[str]:
        """
        Convert a list of relationships from dictionaries to formatted strings
        to be better readable by an llm

        Args:
            triples (List[Dict[str,str]]): a list relationships in the form
                {'start':<from_label>, 'type':<edge_label>, 'end':<from_label>}

        Returns:
            List[str]: a list of relationships in the form
                "(:"<from_label>")-[:"<edge_label>"]->(:"<to_label>")"
        """
        triple_template = '(:"{start}")-[:"{type}"]->(:"{end}")'
        triple_schema = [triple_template.format(**triple) for triple in triples]

        return triple_schema

    def _get_labels(self) -> Tuple[List[str], List[str]]:
        """
        Get all labels of a graph (for both edges and vertices)
        by querying the graph metadata table directly

        Returns
            Tuple[List[str]]: 2 lists, the first containing vertex
                labels and the second containing edge labels
        """

        e_labels_records = self.structured_query(
            """
            SELECT ARRAY(
                    SELECT labname 
                    FROM ag_label 
                    WHERE labkind = 'e' 
                    AND graphid = {}
                    AND labname NOT IN ('ag_edge')
                ) as labels;
            """.format(self.graphid)
        )
        e_labels = e_labels_records[0]["labels"] if e_labels_records else []

        n_labels_records = self.structured_query(
            """
            SELECT labels FROM label_catalog
            WHERE graph_id = {}
            """.format(self.graphid)
        )
        n_labels = n_labels_records[0]["labels"] if n_labels_records else []

        return n_labels, e_labels