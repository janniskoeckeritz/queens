"""Sqlite module."""
import logging
import sqlite3
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pqueens.utils.decorators import safe_operation
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.restructure_data_format import (
    np_array_from_binary,
    np_array_to_binary,
    obj_from_binary,
    obj_to_binary,
    pd_dataframe_from_binary,
    pd_dataframe_to_binary,
)

from .database import Database, QUEENSDatabaseError

# For sqlite the waiting times need to be higher, especially if fast models are used.
safe_sqlitedb_operation = partial(safe_operation, max_number_of_attempts=10, waiting_time=0.1)

_logger = logging.getLogger(__name__)


def sqlite_binary_wrapper(function):
    """Wrap binary output of function to sqlite binary type.

    Args:
        function (fun): Function to be wrapped
    Returns:
        (function) binarized function
    """

    def binarizer(*args, **kwargs):
        binary_out = function(*args, **kwargs)
        return sqlite3.Binary(binary_out)

    return binarizer


def type_to_sqlite(object):
    """Get sqlite type from object.

    Args:
        object: object to be stored in the db

    Returns:
        (str) sqlite data type
    """
    if isinstance(object, str):
        return "TEXT"
    if isinstance(object, int):
        return "INT"
    if isinstance(object, float):
        return "REAL"
    if isinstance(object, xr.DataArray):
        return "XARRAY"
    if isinstance(object, pd.DataFrame):
        return "PDDATAFRAME"
    if isinstance(object, np.ndarray):
        return "NPARRAY"
    if isinstance(object, list):
        return "LIST"
    if isinstance(object, dict):
        return "DICT"


# Add the adapters for different types to sqlite
sqlite3.register_adapter(np.ndarray, sqlite_binary_wrapper(np_array_to_binary))
sqlite3.register_adapter(xr.DataArray, sqlite_binary_wrapper(obj_to_binary))
sqlite3.register_adapter(pd.DataFrame, sqlite_binary_wrapper(pd_dataframe_to_binary))
sqlite3.register_adapter(list, sqlite_binary_wrapper(obj_to_binary))
sqlite3.register_adapter(dict, sqlite_binary_wrapper(obj_to_binary))

# Add the converters, i.e. back to the objects
sqlite3.register_converter("NPARRAY", np_array_from_binary)
sqlite3.register_converter("XARRAY", obj_from_binary)
sqlite3.register_converter("PDDATAFRAME", pd_dataframe_from_binary)
sqlite3.register_converter("LIST", obj_from_binary)
sqlite3.register_converter("DICT", obj_from_binary)


class SQLite(Database):
    """SQLite wrapper for QUEENS."""

    @classmethod
    def from_config_create_database(cls, config):
        """From config create database.

        Args:
            config (dict): Problem description

        Returns:
            sqlite database object
        """
        db_name = config['database'].get('name')
        if not db_name:
            db_name = config['global_settings'].get('experiment_name', 'dummy')

        db_path = config['database'].get('file')
        if db_path is None:
            db_path = Path(config["global_settings"]["output_dir"]).joinpath(db_name + ".sqlite.db")
            _logger.info(
                f"No path for the sqlite database was provided, defaulting to {db_path.resolve()}"
            )
        else:
            db_path = Path(db_path)
        reset_existing_db = config['database'].get('reset_existing_db', True)

        return cls(db_name=db_name, reset_existing_db=reset_existing_db, database_path=db_path)

    def __init__(self, db_name, reset_existing_db, database_path):
        """Initialise database.

        Args:
            db_name (str): Database name
            reset_existing_db (bool): Bool to reset database if desired
            database_path (Pathlib.Path): Path to database object
        """
        super().__init__(db_name, reset_existing_db)
        self.database_path = database_path
        self.existing_tables = {}

    def _check_connection(self):
        """Try to connect to the database.

        Returns:
            True: if sucessfull connection
        """
        try:
            connection = sqlite3.connect(
                self.database_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
            )
            connection.cursor()

            return True
        except Exception as exception:
            raise QUEENSDatabaseError(
                "Could not connect to sqlite database from path {self.database_path}"
            ) from exception

    def _connect(self):
        """Connect to the database.

        Here the connection is checked.
        """
        self._check_connection()

        _logger.info(f"Connected to {self.database_path}")

    @safe_sqlitedb_operation
    def _execute(self, query, commit=False, parameters=None):
        """Execute query.

        Args:
            query (str): Query to be executed
            commit (bool, optional): Commit to the connection. Defaults to False.
            parameters (tuple, optional): Parameters to inject into the query. Defaults to None.

        Returns:
            sqlite.cursor: database cursor
        """
        connection = sqlite3.connect(
            self.database_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        cursor = connection.cursor()
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)
        if commit:
            connection.commit()
        return cursor

    def _disconnect(self):
        """Disconnect the database."""
        _logger.info("Disconnected the database")

    def _get_all_table_names(self):
        """Get all table names in the database file.

        Returns:
            list: Name of tables
        """
        query = "SELECT name FROM sqlite_schema WHERE type='table';"
        cursor = self._execute(query)
        return cursor.fetchall()

    def _delete_table(self, table_name):
        """Delete table by name.

        Args:
            table_name (table_name): Name of the table
        """
        query = f"DROP TABLE {table_name};"
        self._execute(query)

    def _clean_database(self):
        """Delete all tables."""
        table_names = self._get_all_table_names()
        for table_name in table_names:
            self._delete_table(table_name[0])

    def _delete_database(self):
        """Delete database file."""
        self._disconnect()
        # Delete database file
        self.database_path.unlink()

    def _update_tables_if_necessary(self, table_name):
        """Add table if it does not exist.

        Args:
            table_name (str): Table name

        Returns:
            boolean: True if table already existed, False if not
        """
        if table_name in self.existing_tables:
            return self.existing_tables[table_name]
        query = f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        cursor = self._execute(query)
        counter = cursor.fetchone()[0]
        if counter == 0:
            self.existing_tables.update({table_name: {}})
            self._create_table(table_name)
            return False

    def _create_table(self, table_name):
        """Create an empty table.

        Args:
            table_name (str): table_name
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} (id integer PRIMARY KEY)"
        self._execute(query, commit=True)

    def _get_table_column_names(self, table_name):
        """Get names of columns in a table.

        Args:
            table_name (str): Name of the table

        Returns:
            tuple: names of the column
        """
        return self.existing_tables[table_name].keys()

    def _update_columns_if_necessary(self, table_name, column_names, column_types):
        """Add columns if necessary.

        Args:
            table_name (str): Name of the table
            column_names (list): List of column names
            column_types (list): List of data types for the columns
        """
        current_column_names = self._get_table_column_names(table_name)
        if isinstance(column_names, str):
            column_names = [column_names]
        for i, column_name in enumerate(column_names):
            if not column_name in current_column_names and column_name:
                column_type = column_types[i]
                self.existing_tables[table_name].update({column_name: column_type})
                if column_name != "id":
                    query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                    self._execute(query, commit=True)

    def save(self, dictionary, experiment_name, experiment_field, batch, field_filters=None):
        """Save a document to the database.

        Args:
            dictionary (dict):          document to be saved to the db
            experiment_name (string):   experiment the data belongs to
            experiment_field (string):  experiment field data belongs to
            batch (int):                batch the data belongs to
            field_filters (dict):       filter to find appropriate document
                                        to create or update
        Returns:
            bool: is this the result of an acknowledged write operation ?
        """
        table_name = experiment_field
        self._update_tables_if_necessary(table_name)
        save_doc = dictionary.copy()
        save_doc.update({"batch": batch})
        keys = list(save_doc.keys())
        items = list(save_doc.values())
        data_types = list(type_to_sqlite(item) for item in items)
        self._update_columns_if_necessary(table_name, keys, data_types)
        if len(keys) == 1:
            placeholder = "?"
            joint_keys = keys[0]
        else:
            placeholder = "?," * len(keys)
            placeholder = placeholder[:-1]
            joint_keys = ", ".join(list(keys))

        fields_setter = [f"{column_name}=?" for column_name in keys]
        query = f"INSERT INTO {table_name}"
        query += f" ({joint_keys}) VALUES ({placeholder})"
        fields_setter = [f"{column_name}=excluded.{column_name}" for column_name in keys]
        query += f" ON CONFLICT(id) DO UPDATE SET {', '.join(fields_setter)}"
        if field_filters is None:
            query += " WHERE id=excluded.id"
        else:
            filter_conditions = [
                f"{filter_column}=?" for filter_column, filter_item in field_filters.items()
            ]
            query += f" WHERE {' AND '.join(filter_conditions)}"
            items += tuple(field_filters.values())

        self._execute(query, parameters=items, commit=True)

    def load(self, experiment_name, batch, experiment_field, field_filters=None):
        """Load document(s) from the database.

        Decompresses any numpy arrays

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            list: list of documents matching query
        """
        table_name = experiment_field
        if self._update_tables_if_necessary(table_name):
            column_names = self.existing_tables[table_name].keys()
            query = f"SELECT {', '.join(column_names)} FROM {table_name} WHERE BATCH={batch}"
            if field_filters is not None:
                filter_conditions = [
                    f"{filter_column}='{filter_item}'"
                    for filter_column, filter_item in field_filters.items()
                ]
                query += f" AND {' AND '.join(filter_conditions)}"
            cursor = self._execute(query)
            entries = cursor.fetchall()
            entries = self._list_of_entry_dicts(table_name, entries)
            if len(entries) == 1:
                entries = entries[0]
            return entries
        else:
            return None

    def _list_of_entry_dicts(self, table_name, entries):
        column_names = self._get_table_column_names(table_name)
        list_of_dict_entries = []
        for entry in entries:
            entry_dict = dict(zip(column_names, entry))
            # entry_dict.pop("id")
            entry_dict.pop("batch")
            list_of_dict_entries.append(entry_dict)
        return list_of_dict_entries

    def remove(self, experiment_name, experiment_field, batch, field_filters):
        """Remove a list of documents from the database.

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to delete
        """
        table_name = experiment_field
        field_filters.update({"batch": batch})
        filter_conditions = [
            f"{filter_column}='{filter_item}'"
            for filter_column, filter_item in field_filters.items()
        ]
        query = f"DELETE FROM {table_name} WHERE {' AND '.join(filter_conditions)};"
        self._execute(query, commit=True)

    def __str__(self):
        """String function of the sqlite object.

        Returns:
            str: table with information
        """
        print_dict = {"Name": self.db_name, "File": self.database_path.resolve()}
        table = get_str_table("QUEENS SQLite database object wrapper", print_dict)
        return table

    def count_documents(self, experiment_name, batch, experiment_field, field_filters=None):
        """Return number of document(s) in collection.

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            int: number of documents in collection
        """
        table_name = experiment_field

        if self._update_tables_if_necessary(table_name):
            query = f"SELECT COUNT(*) FROM {table_name} WHERE BATCH={batch}"
            if field_filters is not None:
                filter_conditions = [
                    f"{filter_column}='{filter_item}'"
                    for filter_column, filter_item in field_filters.items()
                ]
                query += f" AND {' AND '.join(filter_conditions)}"
            cursor = self._execute(query)
            entries = cursor.fetchall()[0][0]
            return entries
        else:
            return 0
