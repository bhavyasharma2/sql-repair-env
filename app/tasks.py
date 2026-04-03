"""
Task definitions for the SQL Repair environment.

Each task contains:
  - schema_sql    : CREATE TABLE statements
  - seed_sql      : INSERT statements to populate the in-memory DB
  - broken_query  : the query the agent must fix
  - objective     : plain-English description of the correct result
  - ground_truth  : list of dicts representing the expected result rows
  - difficulty    : easy | medium | hard
"""

from typing import Any

TASKS: dict[str, dict[str, Any]] = {

    #TASK 1 — EASY
    # Broken query has two syntax errors:
    #   1. missing comma in SELECT list
    #   2. FORM instead of FROM
    # Any model that can read SQL should fix this immediately.
    # Grader: does the query execute and return rows with the right schema?

    "task_easy": {
        "task_id": "task_easy",
        "difficulty": "easy",
        "objective": (
            "Return each employee's full name and their annual salary "
            "for all employees in the 'Engineering' department, "
            "ordered by salary descending."
        ),
        "schema_sql": """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name  TEXT NOT NULL,
    department TEXT NOT NULL,
    salary     REAL NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO employees VALUES (1, 'Alice', 'Chen',    'Engineering', 120000);
INSERT INTO employees VALUES (2, 'Bob',   'Martin',  'Engineering', 95000);
INSERT INTO employees VALUES (3, 'Carol', 'Singh',   'Marketing',   88000);
INSERT INTO employees VALUES (4, 'David', 'Park',    'Engineering', 140000);
INSERT INTO employees VALUES (5, 'Eve',   'Torres',  'HR',          75000);
""".strip(),
        # Bugs: missing comma between first_name and last_name; FORM instead of FROM
        "broken_query": """
SELECT first_name last_name, salary
FORM employees
WHERE department = 'Engineering'
ORDER BY salary DESC;
""".strip(),
        "ground_truth": [
            {"first_name": "David", "last_name": "Park",   "salary": 140000.0},
            {"first_name": "Alice", "last_name": "Chen",   "salary": 120000.0},
            {"first_name": "Bob",   "last_name": "Martin", "salary": 95000.0},
        ],
    },


    #TASK 2 — MEDIUM
    # Schema: orders + customers + products (3-table join)
    # Broken query uses INNER JOIN where a LEFT JOIN is needed,
    # AND filters on the wrong column alias, causing wrong rows.
    # Agent must understand join semantics to fix this.
    # Grader: do the result rows match ground truth (order + values)?

    "task_medium": {
        "task_id": "task_medium",
        "difficulty": "medium",
        "objective": (
            "For each customer, return their name and the total value of ALL their orders "
            "(including customers who have never ordered — show 0 for them). "
            "Show only customers whose total order value exceeds 200. "
            "Order by total_value descending."
        ),
        "schema_sql": """
CREATE TABLE customers (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount      REAL    NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO customers VALUES (1, 'Priya Sharma');
INSERT INTO customers VALUES (2, 'James Okafor');
INSERT INTO customers VALUES (3, 'Li Wei');
INSERT INTO customers VALUES (4, 'Sara Malik');

INSERT INTO orders VALUES (1, 1, 150.00);
INSERT INTO orders VALUES (2, 1, 200.00);
INSERT INTO orders VALUES (3, 2, 80.00);
INSERT INTO orders VALUES (4, 3, 500.00);
INSERT INTO orders VALUES (5, 3, 120.00);
""".strip(),
        # Bugs:
        #   1. INNER JOIN loses Sara Malik (no orders) — should be LEFT JOIN
        #   2. HAVING filters on amount instead of total_value
        "broken_query": """
SELECT c.name, SUM(o.amount) AS total_value
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
HAVING amount > 200
ORDER BY total_value DESC;
""".strip(),
        "ground_truth": [
            {"name": "Li Wei",       "total_value": 620.0},
            {"name": "Priya Sharma", "total_value": 350.0},
        ],
    },


    #TASK 3 — HARD
    # Schema: sales + products + regions (3-table join with window function)
    # Broken query uses ROW_NUMBER() partitioned by the wrong column —
    # it partitions by product_id instead of region_id, so instead of
    # returning the top-selling product PER REGION it returns one row
    # per product across the entire dataset.
    # The query runs without error — the bug is a window function
    # partitioning mistake, a well-known LLM weak spot.

    "task_hard": {
        "task_id": "task_hard",
        "difficulty": "hard",
        "objective": (
            "For each region, return the name of the top-selling product "
            "(highest total quantity sold) and its total quantity. "
            "If two products tie, return the one with the lower product id. "
            "Order results by region name ascending."
        ),
        "schema_sql": """
CREATE TABLE regions (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE products (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE sales (
    id         INTEGER PRIMARY KEY,
    region_id  INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity   INTEGER NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO regions  VALUES (1, 'East');
INSERT INTO regions  VALUES (2, 'West');

INSERT INTO products VALUES (1, 'Alpha');
INSERT INTO products VALUES (2, 'Beta');
INSERT INTO products VALUES (3, 'Gamma');

-- East: Alpha=30, Beta=50, Gamma=10  → top = Beta(50)
INSERT INTO sales VALUES (1, 1, 1, 30);
INSERT INTO sales VALUES (2, 1, 2, 50);
INSERT INTO sales VALUES (3, 1, 3, 10);

-- West: Alpha=70, Beta=20, Gamma=70  → tie Alpha/Gamma → Alpha wins (lower id)
INSERT INTO sales VALUES (4, 2, 1, 70);
INSERT INTO sales VALUES (5, 2, 2, 20);
INSERT INTO sales VALUES (6, 2, 3, 70);
""".strip(),
        # Bug: PARTITION BY p.id instead of PARTITION BY s.region_id
        # This ranks products globally per product (always rank=1) instead of
        # per region, so the WHERE rn = 1 filter keeps all rows.
        "broken_query": """
WITH ranked AS (
    SELECT
        r.name  AS region_name,
        p.name  AS product_name,
        SUM(s.quantity) AS total_qty,
        ROW_NUMBER() OVER (
            PARTITION BY p.id
            ORDER BY SUM(s.quantity) DESC, p.id ASC
        ) AS rn
    FROM sales s
    JOIN regions  r ON s.region_id  = r.id
    JOIN products p ON s.product_id = p.id
    GROUP BY r.id, r.name, p.id, p.name
)
SELECT region_name, product_name, total_qty
FROM ranked
WHERE rn = 1
ORDER BY region_name ASC;
""".strip(),
        "ground_truth": [
            {"region_name": "East", "product_name": "Beta",  "total_qty": 50},
            {"region_name": "West", "product_name": "Alpha", "total_qty": 70},
        ],
    },


    #TASK 4 — EXPERT
    # Schema: employees table with manager_id (org hierarchy)
    # Broken recursive CTE traverses the org chart but has TWO bugs:
    #   1. The anchor selects WHERE manager_id IS NOT NULL instead of IS NULL
    #      (starts from non-root nodes — wrong starting point)
    #   2. The recursive join uses e.id = cte.id instead of e.manager_id = cte.id
    #      (never actually walks down the tree)
    # The query returns rows but they are completely wrong.
    # This requires understanding recursive CTEs deeply — frontier models
    # often fix only one of the two bugs, scoring partial credit.

    "task_expert": {
        "task_id": "task_expert",
        "difficulty": "expert",
        "objective": (
            "Return every employee's name and their depth in the org chart "
            "(CEO = depth 0, direct reports = depth 1, etc.), "
            "ordered by depth ascending then name ascending."
        ),
        "schema_sql": """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    manager_id INTEGER          -- NULL means this person is the root (CEO)
);
""".strip(),
        "seed_sql": """
-- Depth 0: CEO
INSERT INTO employees VALUES (1, 'Alice',   NULL);
-- Depth 1: Alice's direct reports
INSERT INTO employees VALUES (2, 'Bob',     1);
INSERT INTO employees VALUES (3, 'Carol',   1);
-- Depth 2: Bob's reports
INSERT INTO employees VALUES (4, 'Dave',    2);
INSERT INTO employees VALUES (5, 'Eve',     2);
-- Depth 2: Carol's report
INSERT INTO employees VALUES (6, 'Frank',   3);
""".strip(),
        # Bugs:
        #   1. Anchor: WHERE manager_id IS NOT NULL  →  should be IS NULL
        #   2. Recursive join: e.id = cte.id  →  should be e.manager_id = cte.id
        "broken_query": """
WITH RECURSIVE org AS (
    SELECT id, name, manager_id, 0 AS depth
    FROM employees
    WHERE manager_id IS NOT NULL

    UNION ALL

    SELECT e.id, e.name, e.manager_id, cte.depth + 1
    FROM employees e
    JOIN org cte ON e.id = cte.id
)
SELECT name, depth
FROM org
ORDER BY depth ASC, name ASC;
""".strip(),
        "ground_truth": [
            {"name": "Alice", "depth": 0},
            {"name": "Bob",   "depth": 1},
            {"name": "Carol", "depth": 1},
            {"name": "Dave",  "depth": 2},
            {"name": "Eve",   "depth": 2},
            {"name": "Frank", "depth": 2},
        ],
    },
}
