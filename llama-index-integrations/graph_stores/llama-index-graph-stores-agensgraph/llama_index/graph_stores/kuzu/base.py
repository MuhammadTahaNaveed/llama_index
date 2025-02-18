"""Agensgraph graph store index."""
import logging
import json, re
from typing import Any, Dict, List, Optional, Union, NamedTuple, Pattern, Tuple

from llama_index.core.graph_stores.types import GraphStore
import psycopg2.extras

logger = logging.getLogger(__name__)

flatten_function = """
    CREATE OR REPLACE FUNCTION flatten(input_array jsonb)
    RETURNS jsonb AS $$
    DECLARE                       
        result jsonb := '[]'::jsonb;
        subelement jsonb;
    BEGIN                      
        FOR subelement IN
            SELECT * FROM jsonb_array_elements(input_array)
        LOOP
            result := result || subelement;
        END LOOP;

        RETURN result;
    END;
    $$ LANGUAGE plpgsql;
"""

rel_query = """
    MATCH (start_node)-[r]->(end_node)
    WITH labels(start_node) AS start, type(r) AS relationship_type, labels(end_node) AS endd, keys(r) AS relationship_properties
    UNWIND endd as end_label
    RETURN DISTINCT {start: start[0], type: relationship_type, end: end_label} AS output;
"""

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

class AgensGraphStore(GraphStore):

    # python type mapping for providing readable types to LLM
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
        node_label: str = "Entity",
        create: bool = True
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name
        self.node_label = node_label

        # check that psycopg2 is installed
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Could not import psycopg2 python package. "
                "Please install it with `pip install psycopg2`."
            )

        self.connection = psycopg2.connect(**conf)

        with self._get_cursor() as curs:
            # check if graph with name graph_name exists
            graph_id_query = (
                """SELECT oid as graphid FROM ag_graph WHERE graphname = '{}';""".format(
                    graph_name
                )
            )

            curs.execute(graph_id_query)
            data = curs.fetchone()

            # if graph doesn't exist and create is True, create it
            if data is None:
                if create:
                    create_statement = """
                        CREATE GRAPH {};
                    """.format(graph_name)

                    try:
                        curs.execute(create_statement)
                        self.connection.commit()
                    except psycopg2.Error as e:
                        raise AgensQueryException(
                            {
                                "message": "Could not create the graph",
                                "detail": str(e),
                            }
                        )

                else:
                    raise Exception(
                        (
                            'Graph "{}" does not exist in the database '
                            + 'and "create" is set to False'
                        ).format(graph_name)
                    )

                curs.execute(graph_id_query)
                data = curs.fetchone()

            # store graph id and refresh the schema
            self.graphid = data.graphid

            # set the graph path to the current graph
            graph_path = """SET graph_path = '{}';""".format(self.graph_name)
            curs.execute(graph_path)

            # create the flatten function
            curs.execute(flatten_function)

            self.refresh_schema()
            self.query(
                """
                CREATE VLABEL IF NOT EXISTS "%s";
                CREATE CONSTRAINT ON "%s" ASSERT n.id IS UNIQUE;
                """
                % (self.node_label, self.node_label)
            )

    def _get_cursor(self) -> psycopg2.extras.NamedTupleCursor:
        """
        get cursor and set graph_path to the current graph
        """

        try:
            import psycopg2.extras
        except ImportError as e:
            raise ImportError(
                "Unable to import psycopg2, please install with "
                "`pip install -U psycopg2`."
            ) from e
        cursor = self.connection.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)
        return cursor

    @property
    def client(self) -> Any:
        return self.connection

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:"{}")-[r]->(n2:"{}")
            WHERE n1.id = '{}'
            RETURN type(r), n2.id;
        """.format(self.node_label, self.node_label, subj)

        with self._get_cursor() as curs:
            curs.execute(query)
            rows = curs.fetchall()
        
        result = []
        for row in rows:
            values = []
            for k in row._fields:
                values.append(getattr(row, k))
            result.append(values)

        return result

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+

        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map
        
        subjs = [subj.lower() for subj in subjs]

        query = (
            f"""MATCH p=(n1:"{self.node_label}")-[*1..{depth}]->() """
            f"""WHERE toLower(n1.id) IN {subjs} """
            "UNWIND relationships(p) AS rel "
            "WITH n1.id AS subj, p, collect([type(rel), endNode(rel).id]) AS path "
            f"RETURN subj, collect(DISTINCT flatten(path)) AS flattened_rels LIMIT {limit}"
        )

        data = self.query(query)

        if not data:
            return rel_map

        for record in data:
            rel_map[record["subj"]] = record["flattened_rels"]
        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        query = """
            MERGE (n1:"%s" {{id: '{subj}'}})
            MERGE (n2:"%s" {{id: '{obj}'}})
            MERGE (n1)-[:"%s"]->(n2)
        """

        query = query % (
            self.node_label,
            self.node_label,
            rel.replace(" ", "_").upper(),
        )

        self.query(query.format(subj=subj, obj=obj))

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(subj: str, obj: str, rel: str) -> None:
            query = """
                MATCH (n1:"{}")-[r:"{}"]->(n2:"{}")
                WHERE n1.id = '{}' AND n2.id = '{}'
                DELETE r
            """.format(self.node_label, rel, self.node_label, subj, obj)
            self.query(query)

        def delete_entity(entity: str) -> None:
            query = """
                MATCH (n:"{}")
                WHERE n.id = '{}'
                DELETE n
            """.format(self.node_label, entity)
            self.query(query)

        def check_edges(entity: str) -> bool:
            query = """
                MATCH (n1:"{}")--()
                WHERE n1.id = '{}'
                RETURN count(*)
            """.format(self.node_label, entity)
            data = self.query(query)
            return bool(data[0]["count"])

        delete_rel(subj, obj, rel)
        if not check_edges(subj):
            delete_entity(subj)
        if not check_edges(obj):
            delete_entity(obj)

    def refresh_schema(self) -> None:
        """
        Refresh the graph schema information by updating the available
        labels, relationships, and properties
        """

        # fetch graph schema information
        n_labels, e_labels = self._get_labels()

        node_properties = self._get_node_properties(n_labels)
        edge_properties = self._get_edge_properties(e_labels)
        triple_schema = self._get_triples()

        # update the formatted string representation
        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {self._format_triples(triple_schema)}
        """

        # update the dictionary representation
        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
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
                vertex = AgensGraphStore.vertex_regex.match(v)
                if vertex:
                    label, vertex_id, properties = vertex.groups()
                    properties = json.loads(properties)
                    vertices[str(vertex_id)] = properties

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)

            if isinstance(v, str):
                vertex = AgensGraphStore.vertex_regex.match(v)
                edge = AgensGraphStore.edge_regex.match(v)

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

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
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
        
    def _get_node_properties(self, n_labels: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch a list of available node properties by node label to be used
        as context for an llm

        Args:
            n_labels (List[str]): a list of node labels to filter for

        Returns:
            List[Dict[str, Any]]: a list of node labels and
                their corresponding properties in the form
                "{
                    'labels': <node_label>,
                    'properties': [
                        {
                            'property': <property_name>,
                            'type': <property_type>
                        },...
                        ]
                }"
        """

        # cypher query to fetch properties of a given label
        node_properties_query = """
            MATCH (a:"{n_label}")
            RETURN properties(a) AS props
            LIMIT 100
        """

        node_properties = []
        with self._get_cursor() as curs:
            for label in n_labels:
                q = node_properties_query.format(
                    n_label=label
                )

                try:
                    curs.execute(q)
                except psycopg2.Error as e:
                    raise AgensQueryException(
                        {
                            "message": "Error fetching node properties",
                            "detail": str(e),
                        }
                    )
                data = curs.fetchall()

                # build a set of distinct properties
                s = set({})
                for d in data:
                    for k, v in d.props.items():
                        s.add((k, self.types[type(v).__name__]))

                np = {
                    "properties": [{"property": k, "type": v} for k, v in s],
                    "labels": label,
                }
                node_properties.append(np)

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

                try:
                    curs.execute(q)
                except psycopg2.Error as e:
                    raise AgensQueryException(
                        {
                            "message": "Error fetching edge properties",
                            "detail": str(e),
                        }
                    )
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
        triple_schema = self.query(rel_query)
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

        e_labels_records = self.query(
            """SELECT ARRAY(
                    SELECT labname 
                    FROM ag_label 
                    WHERE labkind = 'e' 
                    AND graphid = {}
                    AND labname NOT IN ('ag_vertex', 'ag_edge')
                ) as labels;
            """.format(self.graphid)
        )
        e_labels = e_labels_records[0]["labels"] if e_labels_records else []

        n_labels_records = self.query(
            """SELECT ARRAY(
                    SELECT labname 
                    FROM ag_label 
                    WHERE labkind = 'v' 
                    AND graphid = {}
                    AND labname NOT IN ('ag_vertex', 'ag_edge')
                ) as labels;
            """.format(self.graphid)
        )
        n_labels = n_labels_records[0]["labels"] if n_labels_records else []

        return n_labels, e_labels