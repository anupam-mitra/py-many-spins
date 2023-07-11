import sqlite3
import pandas

"""This module consist SQL statements and python functions to query a
relational database using the `sqlite3` library. The results are 
returned as `pandas.DataFrame` objects.
"""

def create_parametric_clause(params, prefix=''):
    """Creates a SQL where clause using the
    parameters in the dictionary `params`
    """
    sql_where_clause = prefix + \
        ' and '.join(['%s = :%s' % (k, k) for k in params.keys()])
    return sql_where_clause

##### SELECT SQL statements #####

sql_select_Hamiltonians_clause = \
    '''
    SELECT * 
    FROM Hamiltonians
    '''

sql_select_Approximations_clause = \
    '''
    SELECT * 
    FROM Approximations
    '''

sql_select_SolutionMethods_clause = \
    '''
    SELECT * 
    FROM SolutionMethods
    '''

sql_select_InitialConditions_clause = \
    '''
    SELECT * 
    FROM InitialConditions
    '''

sql_select_Decoherence_clause = \
    '''
    SELECT * 
    FROM Decoherence
    '''

def query_Hamiltonians(params, dbfilename):
    """Query the Hamiltonian table
    """

    sql_where_clause = create_parametric_clause(params, "WHERE")

    sql = sql_select_Hamiltonians_clause + sql_where_clause

    conn = sqlite3.connect(dbfilename)
    result = pandas.read_sql_query(sql, conn, params=params)
    conn.close()

    return result

def query_Approximations(params, dbfilename):
    """Query the Approximations table
    """

    sql_where_clause = create_parametric_clause(params, "WHERE")

    sql = sql_select_Approximations_clause + sql_where_clause

    conn = sqlite3.connect(dbfilename)
    result = pandas.read_sql_query(sql, conn, params=params)
    conn.close()

    return result


def query_SolutionMethods(params, dbfilename):
    """Query the SolutionMethods table
    """

    sql_where_clause = create_parametric_clause(params, "WHERE")

    sql = sql_select_SolutionMethods_clause + sql_where_clause

    conn = sqlite3.connect(dbfilename)
    result = pandas.read_sql_query(sql, conn, params=params)
    conn.close()

    return result

def query_InitialConditions(params, dbfilename):
    """Query the InitialConditions table
    """

    sql_where_clause = create_parametric_clause(params, "WHERE")

    sql = sql_select_InitialConditions_clause + sql_where_clause

    conn = sqlite3.connect(dbfilename)
    result = pandas.read_sql_query(sql, conn, params=params)
    conn.close()

    return result

def query_Decoherence(params, dbfilename):
    """Query the Decoherence table
    """

    sql_where_clause = create_parametric_clause(params, "WHERE")

    sql = sql_select_Decoherence_clause + sql_where_clause

    conn = sqlite3.connect(dbfilename)
    result = pandas.read_sql_query(sql, conn, params=params)
    conn.close()

    return result

 ##### INSERT SQL statements #####

sql_insert_States_clause = \
    '''
    INSERT INTO States
    VALUES
    '''

def insert_States(params, dbfilename):
    """Insert a state
    """

    sql_value_clause = create_parametric_clause(params, "VALUES")

    sql = sql_insert_States_clause + sql_value_clause
    conn = sqlite3.connect(dbfilename)
    conn.execute(sql, params)
    conn.commit()
    conn.close()
