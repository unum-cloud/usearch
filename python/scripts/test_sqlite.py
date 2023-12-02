import sqlite3
import json
import math
import pytest
import numpy as np

import usearch
from usearch.io import load_matrix, save_matrix
from usearch.index import search
from usearch.eval import random_vectors

from usearch.index import Match, Matches, BatchMatches, Index, Indexes


batch_sizes = [1, 3, 20]
dimensions = [3, 97, 256]


@pytest.mark.parametrize("num_vectors", batch_sizes)
@pytest.mark.parametrize("ndim", dimensions)
def test_sqlite_distances_in_high_dimensions(num_vectors: int, ndim: int):
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(usearch.sqlite)

    cursor = conn.cursor()

    # Create a table with additional columns for f32 and f16 BLOBs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_table (
            id INTEGER PRIMARY KEY,
            vector_json JSON,
            vector_f32 BLOB,
            vector_f16 BLOB
        )
    """
    )

    # Generate and insert random vectors
    vectors = []

    for i in range(num_vectors):
        # Generate a random 256-dimensional vector
        vector = np.random.rand(dim)
        vectors.append(vector)

        # Convert the vector to f32 and f16
        vector_f32 = np.float32(vector)
        vector_f16 = np.float16(vector)

        # Insert the vector into the database as JSON and as BLOBs
        cursor.execute(
            """
            INSERT INTO vector_table (vector_json, vector_f32, vector_f16) VALUES (?, ?, ?)
        """,
            (json.dumps(vector.tolist()), vector_f32.tobytes(), vector_f16.tobytes()),
        )

    # Commit changes
    conn.commit()

    similarities = """
    SELECT 
        a.id AS id1,
        b.id AS id2,
        distance_cosine_f32(a.vector_json, b.vector_json) AS cosine_similarity_json,
        distance_cosine_f32(a.vector_f32, b.vector_f32) AS cosine_similarity_f32,
        distance_cosine_f16(a.vector_f16, b.vector_f16) AS cosine_similarity_f16
    FROM 
        vector_table AS a,
        vector_table AS b
    WHERE 
        a.id < b.id;
    """
    cursor.execute(similarities)

    for a, b, similarity_json, similarity_f32, similarity_f16 in cursor.fetchall():
        assert math.isclose(similarity_json, similarity_f32, abs_tol=0.1)
        assert math.isclose(similarity_json, similarity_f16, abs_tol=0.1)

@pytest.mark.parametrize("num_vectors", batch_sizes)
def test_sqlite_distances_in_low_dimensions(num_vectors: int):
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(usearch.sqlite)

    cursor = conn.cursor()

    # Create a table with additional columns for f32 and f16 BLOBs
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vector_table (
            id INTEGER PRIMARY KEY,
            vector_d0 FLOAT,
            vector_d1 FLOAT,
            vector_d2 FLOAT,
            vector_d3 FLOAT,
            description TEXT
        )
    """
    )

    # Generate and insert random vectors
    vectors = []

    for i in range(num_vectors):
        # Generate a random 256-dimensional vector
        vector = np.random.rand(4)
        vectors.append(vector)

        # Insert the vector into the database as scalars
        cursor.execute(
            """
            INSERT INTO vector_table (vector_d0, vector_d1, vector_d2, vector_d3) VALUES (?, ?, ?, ?)
        """,
            (*vector),
        )

    # Commit changes
    conn.commit()

    similarities = """
    SELECT 
        a.id AS id1,
        b.id AS id2,
        distance_cosine_f32(a.vector_d0, a.vector_d1, b.vector_d0, b.vector_d1) AS cosine_similarity_f32,
        distance_cosine_f16(a.vector_d0, a.vector_d1, b.vector_d0, b.vector_d1) AS cosine_similarity_f16,
        distance_haversine_meters(a.vector_d0, a.vector_d1, b.vector_d0, b.vector_d1) AS haversine_meters
    FROM 
        vector_table AS a,
        vector_table AS b
    WHERE 
        a.id < b.id;
    """
    cursor.execute(similarities)

    for a, b, similarity_f32, similarity_f16, haversine_meters in cursor.fetchall():
        # assert math.isclose(similarity_json, similarity_f32, abs_tol=0.1)
        # assert math.isclose(similarity_json, similarity_f16, abs_tol=0.1)
        pass