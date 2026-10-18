"""
Fill the shop tables with fake but realistic data.

Run from the project root (after db/schema.sql is loaded):
    python -m db.seed

The data is the same on every run: every random choice comes from one
random generator started with a fixed seed, and the name lists are in
this file (not from a library whose output can change between versions).
The last line printed is a fingerprint of all rows; run the seed twice
and the fingerprint matches.

Besides the random data, a few "planted" cases are added on purpose,
because the eval questions ask about them:
- Priya Sharma's order on 2026-02-18
- Omar Haddad's order on 2026-05-02 and his damaged-item ticket on 2026-05-09
- five premium customers with a delivered order (Jan 10 - Mar 31) and a
  refund ticket in April, and no other customer matching that pattern
- orders placed in March that are still "processing"
- a customer with two open delivery tickets, and one who bought
  electronics in May and then opened a quality ticket
- no tie for 3rd place in "top 3 products by quantity, April to June"
"""

import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from app.db import connect

SEED = 42
START = date(2026, 1, 1)
END = date(2026, 6, 30)
N_RANDOM_CUSTOMERS = 150

# ---------------------------------------------------------------------------
# Fixed lists
# ---------------------------------------------------------------------------

PRODUCTS = [
    # (name, category, price in USD)
    ("Aurora Noise-Cancelling Headphones", "electronics", "149.00"),
    ("Nimbus Bluetooth Speaker", "electronics", "89.99"),
    ("Blue Ridge Smartwatch", "electronics", "199.00"),
    ("Pulse Wireless Earbuds", "electronics", "79.00"),
    ("Volt Power Bank 20000mAh", "electronics", "39.99"),
    ("Orbit USB-C Hub", "electronics", "45.00"),
    ("Echo Mechanical Keyboard", "electronics", "119.00"),
    ("Swift Wireless Mouse", "electronics", "29.99"),
    ("Atlas 27-inch Monitor", "electronics", "279.00"),
    ("Glass Food Containers (Set of 4)", "home", "24.99"),
    ("Linen Throw Blanket", "home", "54.00"),
    ("Ceramic Pour-Over Coffee Set", "home", "42.00"),
    ("Bamboo Cutting Board", "home", "19.50"),
    ("Cast Iron Skillet 10-inch", "home", "38.00"),
    ("Aroma Diffuser", "home", "31.00"),
    ("Lumen LED Desk Lamp", "home", "34.50"),
    ("Everyday Canvas Tote", "apparel", "24.50"),
    ("Merino Wool Beanie", "apparel", "28.00"),
    ("Trail Running Socks (3 Pack)", "apparel", "18.00"),
    ("Packable Rain Jacket", "apparel", "89.00"),
    ("Organic Cotton Hoodie", "apparel", "64.00"),
    ("Leather Card Wallet", "apparel", "35.00"),
    ("The Quiet Garden (Hardcover)", "books", "22.00"),
    ("Field Guide to Himalayan Birds", "books", "32.00"),
    ("Python for Busy People", "books", "39.00"),
    ("Night Market Cookbook", "books", "27.50"),
    ("Cork Yoga Mat", "sports", "58.00"),
    ("Adjustable Dumbbell Pair", "sports", "129.00"),
    ("Insulated Water Bottle 1L", "sports", "26.00"),
    ("Resistance Band Set", "sports", "21.00"),
    ("Trekking Poles (Pair)", "sports", "49.00"),
    ("Shea Butter Hand Cream", "beauty", "12.00"),
    ("Charcoal Face Wash", "beauty", "15.00"),
    ("Vitamin C Serum", "beauty", "29.00"),
]

CITIES = [
    # (city, country, weight)
    ("Kathmandu", "Nepal", 25),
    ("Lalitpur", "Nepal", 8),
    ("Pokhara", "Nepal", 7),
    ("London", "United Kingdom", 10),
    ("Manchester", "United Kingdom", 4),
    ("Toronto", "Canada", 8),
    ("Vancouver", "Canada", 4),
    ("Sydney", "Australia", 7),
    ("Melbourne", "Australia", 4),
    ("Mumbai", "India", 6),
    ("Bengaluru", "India", 5),
    ("New York", "United States", 6),
    ("Austin", "United States", 3),
    ("Berlin", "Germany", 3),
]

FIRST_NAMES = [
    "Aarav",
    "Sita",
    "Bibek",
    "Pooja",
    "Sanjay",
    "Asmita",
    "Nabin",
    "Kritika",
    "Suman",
    "Rajesh",
    "Anisha",
    "Prakash",
    "Sabina",
    "Dipesh",
    "Manisha",
    "Kiran",
    "Sneha",
    "James",
    "Emily",
    "Oliver",
    "Charlotte",
    "Harry",
    "Amelia",
    "George",
    "Isla",
    "Liam",
    "Olivia",
    "Noah",
    "Emma",
    "Lucas",
    "Chloe",
    "Ethan",
    "Zoe",
    "Mason",
    "Arjun",
    "Ananya",
    "Vikram",
    "Diya",
    "Rahul",
    "Ishaan",
    "Meera",
    "Kabir",
    "Wei",
    "Hana",
    "Yuki",
    "Min-jun",
    "Sara",
    "Lukas",
    "Lena",
    "Fatima",
    "Yusuf",
    "Layla",
    "Mateo",
    "Sofia",
    "Ava",
    "Jack",
    "Ruby",
    "Leo",
    "Nora",
]

LAST_NAMES = [
    "Shrestha",
    "Adhikari",
    "Thapa",
    "Karki",
    "Rai",
    "Magar",
    "Basnet",
    "Poudel",
    "Koirala",
    "Bhattarai",
    "KC",
    "Joshi",
    "Smith",
    "Jones",
    "Taylor",
    "Brown",
    "Wilson",
    "Evans",
    "Walker",
    "Wright",
    "Martin",
    "Tremblay",
    "Roy",
    "Campbell",
    "Patel",
    "Iyer",
    "Reddy",
    "Nair",
    "Kapoor",
    "Singh",
    "Kim",
    "Nguyen",
    "Tanaka",
    "Wang",
    "Muller",
    "Schmidt",
    "Garcia",
    "Lopez",
    "Ali",
    "Khan",
    "Murphy",
    "Kelly",
]

TICKET_TEXT = {
    "delivery": [
        "My package hasn't arrived yet and tracking hasn't updated in days.",
        "The courier marked the order as delivered but nothing came.",
        "Can you tell me when my order will ship? It still says processing.",
        "Delivery is taking much longer than the estimate I was given.",
    ],
    "refund": [
        "I returned the item two weeks ago and haven't seen my refund.",
        "I'd like a refund, the product isn't what I expected.",
        "My order was cancelled but I was still charged.",
        "Please refund the order, it no longer fits my needs.",
    ],
    "damaged_item": [
        "The item arrived broken, the box was crushed.",
        "One of the pieces was cracked when I opened the package.",
        "The product arrived scratched and dented.",
    ],
    "quality": [
        "It stopped working after a few weeks of normal use.",
        "The material feels much cheaper than in the photos.",
        "The stitching came apart after one wash.",
        "Battery life is far below what was advertised.",
    ],
}

# People added on purpose. They are kept out of the random name pool so
# their names stay unique.
PLANTED_NAMES = {
    "Priya Sharma",
    "Omar Haddad",
    "Anjali Gurung",
    "Daniel Brooks",
    "Mei Lin Chen",
    "Rohan Mehta",
    "Sophie Laurent",
    "Bikash Tamang",
    "Grace Okafor",
}


# ---------------------------------------------------------------------------
# In-memory rows
# ---------------------------------------------------------------------------


@dataclass
class Shop:
    customers: list = field(default_factory=list)  # (id, name, email, city, country, plan, signup)
    orders: list = field(default_factory=list)  # [id, customer_id, date, status, total]
    items: list = field(default_factory=list)  # (id, order_id, product_id, qty, unit_price)
    tickets: list = field(default_factory=list)  # [id, customer_id, order_id, topic, status, date, text]

    def add_customer(self, name, city, country, plan, signup):
        cid = len(self.customers) + 1
        email = f"{name.lower().replace(' ', '.').replace('-', '')}.{cid}@example.com"
        self.customers.append((cid, name, email, city, country, plan, signup))
        return cid

    def add_order(self, customer_id, order_date, status, lines):
        """lines: list of (product_id, quantity)."""
        oid = len(self.orders) + 1
        total = Decimal("0.00")
        for product_id, qty in lines:
            price = Decimal(PRODUCTS[product_id - 1][2])
            self.items.append((len(self.items) + 1, oid, product_id, qty, price))
            total += price * qty
        self.orders.append([oid, customer_id, order_date, status, total])
        return oid

    def add_ticket(self, customer_id, order_id, topic, status, created, text):
        tid = len(self.tickets) + 1
        self.tickets.append([tid, customer_id, order_id, topic, status, created, text])
        return tid


def product_id(name: str) -> int:
    for i, (pname, _, _) in enumerate(PRODUCTS, start=1):
        if pname == name:
            return i
    raise KeyError(name)


def random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_random(shop: Shop, rng: random.Random) -> None:
    cities = [(c, k) for c, k, _ in CITIES]
    weights = [w for _, _, w in CITIES]
    used = set(PLANTED_NAMES)

    for _ in range(N_RANDOM_CUSTOMERS):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used:
                used.add(name)
                break
        city, country = rng.choices(cities, weights)[0]
        plan = "premium" if rng.random() < 0.3 else "basic"
        signup = random_date(rng, date(2025, 7, 1), date(2026, 6, 20))
        shop.add_customer(name, city, country, plan, signup)

    for cid, *_rest in list(shop.customers):
        signup = shop.customers[cid - 1][6]
        n_orders = rng.choices(range(8), weights=[10, 15, 20, 20, 15, 10, 6, 4])[0]
        first_day = max(signup, START)
        for _ in range(n_orders):
            order_date = random_date(rng, first_day, END)
            n_lines = rng.choices([1, 2, 3], weights=[55, 30, 15])[0]
            picks = rng.sample(range(1, len(PRODUCTS) + 1), n_lines)
            lines = [(p, rng.choices([1, 2, 3], weights=[75, 20, 5])[0]) for p in picks]

            r = rng.random()
            if r < 0.08:
                status = "cancelled"
            elif order_date >= date(2026, 6, 24):
                status = "processing" if rng.random() < 0.8 else "delivered"
            else:
                status = "processing" if rng.random() < 0.02 else "delivered"
            shop.add_order(cid, order_date, status, lines)

    for oid, cid, order_date, status, _total in list(shop.orders):
        if status == "cancelled":
            if rng.random() >= 0.3:
                continue
            topic = "refund"
        else:
            if rng.random() >= 0.18:
                continue
            if status == "processing":
                topic = "delivery"
            else:
                topic = rng.choices(
                    ["delivery", "refund", "damaged_item", "quality"], weights=[35, 30, 15, 20]
                )[0]
        created = min(order_date + timedelta(days=rng.randint(1, 21)), END)
        recent = created >= date(2026, 6, 16)
        open_chance = 0.7 if recent else 0.15
        ticket_status = "open" if rng.random() < open_chance else "closed"
        shop.add_ticket(cid, oid, topic, ticket_status, created, rng.choice(TICKET_TEXT[topic]))


def remove_accidental_refund_matches(shop: Shop) -> None:
    """Keep the planted refund question's answer exactly the five planted people.

    A random premium customer with a delivered order (Jan 10 - Mar 31) and a
    refund ticket in April would also match, so those April refund tickets
    become delivery tickets instead.
    """
    premium = {c[0] for c in shop.customers if c[5] == "premium"}
    early_delivered = {
        o[1] for o in shop.orders if o[3] == "delivered" and date(2026, 1, 10) <= o[2] <= date(2026, 3, 31)
    }
    for t in shop.tickets:
        _tid, cid, _oid, topic, _status, created, _text = t
        april = date(2026, 4, 1) <= created <= date(2026, 4, 30)
        if topic == "refund" and april and cid in premium and cid in early_delivered:
            t[3] = "delivery"
            t[6] = TICKET_TEXT["delivery"][0]


def add_planted(shop: Shop, rng: random.Random) -> None:
    pid = product_id

    # Priya Sharma: the Feb 18 order (MIXED question, currency conversion).
    priya = shop.add_customer("Priya Sharma", "Kathmandu", "Nepal", "premium", date(2025, 11, 3))
    shop.add_order(
        priya,
        date(2026, 2, 18),
        "delivered",
        [(pid("Aurora Noise-Cancelling Headphones"), 1), (pid("Everyday Canvas Tote"), 1)],
    )
    shop.add_order(priya, date(2026, 4, 7), "delivered", [(pid("Insulated Water Bottle 1L"), 2)])

    # Omar Haddad: the May 2 order and the damaged-item ticket on May 9.
    omar = shop.add_customer("Omar Haddad", "Toronto", "Canada", "basic", date(2025, 12, 12))
    shop.add_order(omar, date(2026, 3, 14), "delivered", [(pid("Cork Yoga Mat"), 1)])
    omar_order = shop.add_order(
        omar,
        date(2026, 5, 2),
        "delivered",
        [(pid("Nimbus Bluetooth Speaker"), 1), (pid("Glass Food Containers (Set of 4)"), 1)],
    )
    shop.add_ticket(
        omar,
        omar_order,
        "damaged_item",
        "open",
        date(2026, 5, 9),
        "The Bluetooth speaker arrived with a cracked grille and the box was crushed.",
    )

    # Five premium customers: delivered order Jan 10 - Mar 31, refund ticket in April.
    refund_cases = [
        ("Anjali Gurung", "Pokhara", "Nepal", date(2026, 2, 3), "Linen Throw Blanket", date(2026, 4, 4)),
        (
            "Daniel Brooks",
            "London",
            "United Kingdom",
            date(2026, 1, 22),
            "Echo Mechanical Keyboard",
            date(2026, 4, 11),
        ),
        ("Mei Lin Chen", "Vancouver", "Canada", date(2026, 3, 9), "Packable Rain Jacket", date(2026, 4, 15)),
        ("Rohan Mehta", "Mumbai", "India", date(2026, 2, 27), "Adjustable Dumbbell Pair", date(2026, 4, 22)),
        ("Sophie Laurent", "Berlin", "Germany", date(2026, 3, 18), "Vitamin C Serum", date(2026, 4, 28)),
    ]
    for name, city, country, order_day, product, refund_day in refund_cases:
        cid = shop.add_customer(name, city, country, "premium", date(2025, 10, 1))
        oid = shop.add_order(cid, order_day, "delivered", [(pid(product), 1)])
        shop.add_ticket(
            cid, oid, "refund", "closed", refund_day, "I'd like a refund, the product isn't what I expected."
        )

    # Two open delivery tickets for one customer (Feb - May).
    bikash = shop.add_customer("Bikash Tamang", "Kathmandu", "Nepal", "basic", date(2025, 9, 15))
    o1 = shop.add_order(bikash, date(2026, 2, 25), "delivered", [(pid("Trekking Poles (Pair)"), 1)])
    o2 = shop.add_order(bikash, date(2026, 4, 12), "delivered", [(pid("Merino Wool Beanie"), 2)])
    shop.add_ticket(bikash, o1, "delivery", "open", date(2026, 3, 3), TICKET_TEXT["delivery"][1])
    shop.add_ticket(bikash, o2, "delivery", "open", date(2026, 4, 19), TICKET_TEXT["delivery"][3])

    # Electronics bought in May, then a quality ticket in June.
    grace = shop.add_customer("Grace Okafor", "London", "United Kingdom", "premium", date(2025, 8, 20))
    g_order = shop.add_order(grace, date(2026, 5, 12), "delivered", [(pid("Pulse Wireless Earbuds"), 1)])
    shop.add_ticket(
        grace,
        g_order,
        "quality",
        "open",
        date(2026, 6, 5),
        "The left earbud stopped charging after three weeks.",
    )

    # Three March orders that are still "processing" (stuck orders).
    random_ids = [c[0] for c in shop.customers[:N_RANDOM_CUSTOMERS] if c[6] <= date(2026, 3, 1)]
    for day in (5, 17, 26):
        cid = rng.choice(random_ids)
        shop.add_order(cid, date(2026, 3, day), "processing", [(rng.randint(1, len(PRODUCTS)), 1)])


def break_third_place_tie(shop: Shop, rng: random.Random) -> None:
    """Make sure 3rd and 4th place differ in "top 3 products, April to June"."""
    while True:
        status_by_order = {o[0]: (o[2], o[3]) for o in shop.orders}
        qty = Counter()
        for _iid, oid, prod, q, _price in shop.items:
            order_date, status = status_by_order[oid]
            if status != "cancelled" and date(2026, 4, 1) <= order_date <= date(2026, 6, 30):
                qty[prod] += q
        ranked = qty.most_common()
        if len(ranked) < 4 or ranked[2][1] != ranked[3][1]:
            return
        # Give the current 3rd place one more sale in May.
        pool = shop.customers[:N_RANDOM_CUSTOMERS]  # never touch the planted people
        cid = rng.choice([c[0] for c in pool if c[6] <= date(2026, 5, 1)])
        shop.add_order(cid, date(2026, 5, rng.randint(1, 28)), "delivered", [(ranked[2][0], 1)])


def build() -> Shop:
    rng = random.Random(SEED)
    shop = Shop()
    build_random(shop, rng)
    remove_accidental_refund_matches(shop)
    add_planted(shop, rng)
    break_third_place_tie(shop, rng)
    return shop


# ---------------------------------------------------------------------------
# Writing to Postgres
# ---------------------------------------------------------------------------

FINGERPRINT_SQL = """
SELECT md5(
    (SELECT string_agg(t::text, '|' ORDER BY id) FROM customers t) ||
    (SELECT string_agg(t::text, '|' ORDER BY id) FROM products t) ||
    (SELECT string_agg(t::text, '|' ORDER BY id) FROM orders t) ||
    (SELECT string_agg(t::text, '|' ORDER BY id) FROM order_items t) ||
    (SELECT string_agg(t::text, '|' ORDER BY id) FROM support_tickets t)
)
"""


def write(shop: Shop, conn) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE support_tickets, order_items, orders, products, customers RESTART IDENTITY CASCADE"
        )
        cur.executemany(
            "INSERT INTO products (id, name, category, price) VALUES (%s, %s, %s, %s)",
            [(i, n, c, Decimal(p)) for i, (n, c, p) in enumerate(PRODUCTS, start=1)],
        )
        cur.executemany(
            "INSERT INTO customers (id, full_name, email, city, country, plan, signup_date)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)",
            shop.customers,
        )
        cur.executemany(
            "INSERT INTO orders (id, customer_id, order_date, status, total) VALUES (%s, %s, %s, %s, %s)",
            [tuple(o) for o in shop.orders],
        )
        cur.executemany(
            "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price)"
            " VALUES (%s, %s, %s, %s, %s)",
            shop.items,
        )
        cur.executemany(
            "INSERT INTO support_tickets (id, customer_id, order_id, topic, status, created_at, description)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [tuple(t) for t in shop.tickets],
        )
        cur.execute(FINGERPRINT_SQL)
        fingerprint = cur.fetchone()[0]
    conn.commit()
    return fingerprint


def main() -> None:
    shop = build()
    with connect() as conn:
        fingerprint = write(shop, conn)
    print(
        f"Seeded {len(shop.customers)} customers, {len(PRODUCTS)} products, "
        f"{len(shop.orders)} orders, {len(shop.items)} order items, {len(shop.tickets)} tickets."
    )
    print(f"Fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
