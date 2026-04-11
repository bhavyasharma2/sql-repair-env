"""
Task definitions for SQL Repair Env.

6 tasks of increasing difficulty covering real-world SQL bug patterns:

  1. easy      — syntax error (FORM instead of FROM, missing comma)
  2. medium    — wrong JOIN type (INNER vs LEFT, HAVING on wrong alias)
  3. hard      — window function PARTITION BY wrong column
  4. expert    — recursive CTE with two bugs
  5. adversarial — aggregation pushed inside WHERE instead of HAVING
  6. business   — multi-CTE with incorrect date truncation logic

Each task has:
  - schema_sql   : CREATE TABLE statements
  - seed_sql     : INSERT rows for a deterministic in-memory DB
  - broken_query : the query the agent must fix
  - objective    : plain-English description of correct result
  - ground_truth : expected rows as list of dicts
  - difficulty   : easy | medium | hard | expert | adversarial | business
  - bug_type     : one-line description of the bug category
"""

from typing import Any

TASKS: dict[str, dict[str, Any]] = {

    # ── 1. EASY — Syntax error ─────────────────────────────────────────────────
    "task_easy": {
        "task_id": "task_easy",
        "difficulty": "easy",
        "bug_type": "syntax error (typo + missing comma)",
        "objective": (
            "Return each Engineering employee's first name, last name, and "
            "annual salary, ordered by salary descending."
        ),
        "schema_sql": """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    first_name TEXT    NOT NULL,
    last_name  TEXT    NOT NULL,
    department TEXT    NOT NULL,
    salary     REAL    NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO employees VALUES (1, 'Alice', 'Chen',   'Engineering', 120000);
INSERT INTO employees VALUES (2, 'Bob',   'Martin', 'Engineering',  95000);
INSERT INTO employees VALUES (3, 'Carol', 'Singh',  'Marketing',    88000);
INSERT INTO employees VALUES (4, 'David', 'Park',   'Engineering', 140000);
INSERT INTO employees VALUES (5, 'Eve',   'Torres', 'HR',           75000);
""".strip(),
        # Bugs: FORM instead of FROM; missing comma between first_name and last_name
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

    # ── 2. MEDIUM — Wrong JOIN + HAVING alias ──────────────────────────────────
    "task_medium": {
        "task_id": "task_medium",
        "difficulty": "medium",
        "bug_type": "wrong JOIN type + HAVING references wrong column",
        "objective": (
            "For each customer, return their name and total order value. "
            "Include customers with no orders (show 0). "
            "Return only customers whose total exceeds 200, ordered by total descending."
        ),
        "schema_sql": """
CREATE TABLE customers (
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL
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
INSERT INTO orders VALUES (3, 2,  80.00);
INSERT INTO orders VALUES (4, 3, 500.00);
INSERT INTO orders VALUES (5, 3, 120.00);
""".strip(),
        # Bugs: INNER JOIN loses Sara Malik; HAVING uses `amount` not `total_value`
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

    # ── 3. HARD — Window function PARTITION BY wrong column ───────────────────
    "task_hard": {
        "task_id": "task_hard",
        "difficulty": "hard",
        "bug_type": "ROW_NUMBER() partitioned by wrong column (silent semantic error)",
        "objective": (
            "For each region, return the top-selling product (highest total quantity) "
            "and its total quantity sold. Break ties by lower product id. "
            "Order results by region name ascending."
        ),
        "schema_sql": """
CREATE TABLE regions (
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL
);
CREATE TABLE products (
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL
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
INSERT INTO sales VALUES (1, 1, 1, 30);
INSERT INTO sales VALUES (2, 1, 2, 50);
INSERT INTO sales VALUES (3, 1, 3, 10);
INSERT INTO sales VALUES (4, 2, 1, 70);
INSERT INTO sales VALUES (5, 2, 2, 20);
INSERT INTO sales VALUES (6, 2, 3, 70);
""".strip(),
        # Bug: PARTITION BY p.id instead of PARTITION BY s.region_id
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
    GROUP BY s.region_id, r.id, r.name, p.id, p.name
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

    # ── 4. EXPERT — Recursive CTE with two bugs ───────────────────────────────
    "task_expert": {
        "task_id": "task_expert",
        "difficulty": "expert",
        "bug_type": "recursive CTE — wrong anchor condition + wrong recursive join",
        "objective": (
            "Return every employee's name and their depth in the org chart "
            "(CEO = depth 0, direct reports = depth 1, etc.), "
            "ordered by depth then name."
        ),
        "schema_sql": """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    manager_id INTEGER
);
""".strip(),
        "seed_sql": """
INSERT INTO employees VALUES (1, 'Alice',  NULL);
INSERT INTO employees VALUES (2, 'Bob',    1);
INSERT INTO employees VALUES (3, 'Carol',  1);
INSERT INTO employees VALUES (4, 'Dave',   2);
INSERT INTO employees VALUES (5, 'Eve',    2);
INSERT INTO employees VALUES (6, 'Frank',  3);
""".strip(),
        # Bugs: anchor IS NOT NULL should be IS NULL; recursive join e.id = cte.id should be e.manager_id = cte.id
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

    # ── 5. ADVERSARIAL — Aggregation in WHERE instead of HAVING ───────────────
    "task_adversarial": {
        "task_id": "task_adversarial",
        "difficulty": "adversarial",
        "bug_type": "aggregate function used in WHERE clause instead of HAVING",
        "objective": (
            "Return each product category and its total revenue (price * quantity), "
            "but only for categories where total revenue exceeds 5000. "
            "Order by total revenue descending."
        ),
        "schema_sql": """
CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,
    price    REAL    NOT NULL
);
CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    quantity   INTEGER NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO products VALUES (1, 'Laptop',   'Electronics', 999.99);
INSERT INTO products VALUES (2, 'Phone',    'Electronics', 699.99);
INSERT INTO products VALUES (3, 'Desk',     'Furniture',   299.99);
INSERT INTO products VALUES (4, 'Chair',    'Furniture',   199.99);
INSERT INTO products VALUES (5, 'Notebook', 'Stationery',    4.99);
INSERT INTO products VALUES (6, 'Pen',      'Stationery',    1.99);
INSERT INTO order_items VALUES (1,  1, 5);
INSERT INTO order_items VALUES (2,  2, 8);
INSERT INTO order_items VALUES (3,  3, 10);
INSERT INTO order_items VALUES (4,  4, 15);
INSERT INTO order_items VALUES (5,  5, 200);
INSERT INTO order_items VALUES (6,  6, 500);
""".strip(),
        # Bug: WHERE SUM(...) > 5000 — aggregate not allowed in WHERE
        "broken_query": """
SELECT p.category, SUM(p.price * oi.quantity) AS total_revenue
FROM products p
JOIN order_items oi ON p.id = oi.product_id
WHERE SUM(p.price * oi.quantity) > 5000
GROUP BY p.category
ORDER BY total_revenue DESC;
""".strip(),
        "ground_truth": [
            {"category": "Electronics", "total_revenue": 10599.87},
            {"category": "Furniture",   "total_revenue": 5999.75},
        ],
    },

    # ── 6. BUSINESS — Multi-CTE with wrong date truncation ────────────────────
    "task_business": {
        "task_id": "task_business",
        "difficulty": "business",
        "bug_type": "wrong date truncation unit in multi-CTE revenue analysis",
        "objective": (
            "Return monthly revenue totals (year-month) for 2024, "
            "ordering by month ascending. "
            "Each row: year_month (format YYYY-MM), total_revenue."
        ),
        "schema_sql": """
CREATE TABLE transactions (
    id         INTEGER PRIMARY KEY,
    amount     REAL    NOT NULL,
    created_at TEXT    NOT NULL
);
""".strip(),
        "seed_sql": """
INSERT INTO transactions VALUES (1,  500.00, '2024-01-05');
INSERT INTO transactions VALUES (2,  750.00, '2024-01-20');
INSERT INTO transactions VALUES (3, 1200.00, '2024-02-10');
INSERT INTO transactions VALUES (4,  300.00, '2024-02-28');
INSERT INTO transactions VALUES (5,  900.00, '2024-03-15');
INSERT INTO transactions VALUES (6,  450.00, '2024-03-22');
INSERT INTO transactions VALUES (7,  100.00, '2023-12-31');
""".strip(),
        # Bug: strftime('%Y', ...) truncates to year, not month — should be '%Y-%m'
        "broken_query": """
WITH monthly AS (
    SELECT
        strftime('%Y', created_at) AS year_month,
        SUM(amount) AS total_revenue
    FROM transactions
    WHERE created_at >= '2024-01-01'
      AND created_at <  '2025-01-01'
    GROUP BY strftime('%Y', created_at)
)
SELECT year_month, total_revenue
FROM monthly
ORDER BY year_month ASC;
""".strip(),
        "ground_truth": [
            {"year_month": "2024-01", "total_revenue": 1250.0},
            {"year_month": "2024-02", "total_revenue": 1500.0},
            {"year_month": "2024-03", "total_revenue": 1350.0},
        ],
    },
}
