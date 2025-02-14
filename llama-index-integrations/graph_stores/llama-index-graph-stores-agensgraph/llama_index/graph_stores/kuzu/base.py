"""Agensgraph graph store index."""

from typing import Any, Dict, List, Optional, Union

from llama_index.core.graph_stores.types import GraphStore
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

class AgensGraphStore(GraphStore):
    def __init__(
        self, graph_name: str, conf: Dict[str, Any], create: bool = True
    ) -> None:
        """Create a new Agensgraph Graph instance."""

        self.graph_name = graph_name

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
            MATCH (n1)-[r]->(n2)
            WHERE n1."ID" = {}
            RETURN r.predicate, n2.ID;
        """.format(subj)

        with self._get_cursor() as curs:
            curs.execute(query)
            rows = curs.fetchall()
        retval = []
        for row in rows:
            # AGS: add our parsing here
            retval.append([row[0], row[1]])
        return retval

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get depth-aware rel map."""
        rel_wildcard = "r:%s*1..%d" % (self.rel_table_name, depth)
        match_clause = "MATCH (n1)-[{}]->(n2)".format(rel_wildcard)
        return_clause = "RETURN n1, r, n2 LIMIT {}".format(limit)

        # AGS: Research usage of params here
        params = []
        if subjs is not None:
            for i, curr_subj in enumerate(subjs):
                if i == 0:
                    where_clause = 'WHERE n1."ID" = {}'.format(i)
                else:
                    where_clause += ' OR n1."ID" = {}'.format(i)
                params.append((str(i), curr_subj))
        else:
            where_clause = ""
        query = f"{match_clause} {where_clause} {return_clause}"

        if subjs is not None:
            # AGS: Research usage of params here
            query_result = self.connection.execute(prepared_statement, dict(params))
        else:
            with self._get_cursor() as curs:
                curs.execute(query)
                rows = curs.fetchall()

        retval: Dict[str, List[List[str]]] = {}
        for row in rows:
            curr_path = []

            #AGS: add our parsing here
            subj = row[0]
            recursive_rel = row[1]
            obj = row[2]
            nodes_map = {}
            nodes_map[(subj["_id"]["table"], subj["_id"]["offset"])] = subj["ID"]
            nodes_map[(obj["_id"]["table"], obj["_id"]["offset"])] = obj["ID"]
            for node in recursive_rel["_nodes"]:
                nodes_map[(node["_id"]["table"], node["_id"]["offset"])] = node["ID"]
            for rel in recursive_rel["_rels"]:
                predicate = rel["predicate"]
                curr_subj_id = nodes_map[(rel["_src"]["table"], rel["_src"]["offset"])]
                curr_path.append(curr_subj_id)
                curr_path.append(predicate)
            # Add the last node
            curr_path.append(obj["ID"])
            if subj["ID"] not in retval:
                retval[subj["ID"]] = []
            retval[subj["ID"]].append(curr_path)
        return retval

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""

        def check_entity_exists(connection: Any, entity: str) -> bool:
            is_exists_query = 'MATCH (n) WHERE n."ID" = {} RETURN n."ID"'.format(entity)
            with self._get_cursor() as curs:
                curs.execute(is_exists_query)
                return curs.fetchone() is not None

        def create_entity(connection: Any, entity: str) -> None:
            create_entity_query = 'CREATE (n: {{"ID": {}}})'.format(entity)
            with self._get_cursor() as curs:
                curs.execute(create_entity_query)

        def check_rel_exists(connection: Any, subj: str, obj: str, rel: str) -> bool:
            is_exists_query = (
                'MATCH (n1)-[r]->(n2) WHERE n1."ID" = {} AND n2."ID" = {} AND r.predicate = "{}" RETURN r.predicate'
            ).format(subj, obj, rel)
            with self._get_cursor() as curs:
                curs.execute(is_exists_query)
                return curs.fetchone() is not None
            
        def create_rel(connection: Any, subj: str, obj: str, rel: str) -> None:
            create_rel_query = (
                'MATCH (n1), (n2) WHERE n1."ID" = {} AND n2."ID" = {} CREATE (n1)-[r:{{"predicate": "{}"}}]->(n2)'
            ).format(subj, obj, rel)
            with self._get_cursor() as curs:
                curs.execute(create_rel_query)

        is_subj_exists = check_entity_exists(self.connection, subj)
        is_obj_exists = check_entity_exists(self.connection, obj)

        if not is_subj_exists:
            create_entity(self.connection, subj)
        if not is_obj_exists:
            create_entity(self.connection, obj)

        if is_subj_exists and is_obj_exists:
            is_rel_exists = check_rel_exists(self.connection, subj, obj, rel)
            if is_rel_exists:
                return

        create_rel(self.connection, subj, obj, rel)

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(connection: Any, subj: str, obj: str, rel: str) -> None:
            delete_rel_query = (
                'MATCH (n1)-[r]->(n2) WHERE n1."ID" = {} AND n2."ID" = {} AND r.predicate = "{}" DELETE r'
            ).format(subj, obj, rel)
            with self._get_cursor() as curs:
                curs.execute(delete_rel_query)

        def delete_entity(connection: Any, entity: str) -> None:
            delete_entity_query = 'MATCH (n) WHERE n."ID" = {} DELETE n'.format(entity)
            with self._get_cursor() as curs:
                curs.execute(delete_entity_query)

        def check_edges(connection: Any, entity: str) -> bool:
            check_edges_query = 'MATCH (n1)-[r]-(n2) WHERE n2."ID" = {} RETURN r.predicate'.format(entity)
            with self._get_cursor() as curs:
                curs.execute(check_edges_query)
                return curs.fetchone() is not None

        delete_rel(self.connection, subj, obj, rel)
        if not check_edges(self.connection, subj):
            delete_entity(self.connection, subj)
        if not check_edges(self.connection, obj):
            delete_entity(self.connection, obj)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        node_table_name: str = "entity",
        rel_table_name: str = "links",
    ) -> "AgensGraphStore":
        """Load from persist dir."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Could not import psycopg2 python package. "
                "Please install it with `pip install psycopg2`."
            )
        # AGS: research cls
        # database = kuzu.Database(persist_dir)
        # return cls(database, node_table_name, rel_table_name)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgensGraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)
