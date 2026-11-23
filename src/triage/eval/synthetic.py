"""Generate the synthetic (unseen) prompts: invented documents, contracts and code specs.

Every prompt is built from templates written for this project and filled with invented names, numbers and
dates, so no model can have seen it in training. Each answer is computed by the script that planted it.
Prompts copy the exact instruction wording of the public items they mirror (SQuAD, CUAD, the handwritten
CX items, HumanEval, GSM8K, Banking77), so the only difference from the public set is seen vs unseen.

  uv run python -m triage.eval.synthetic     writes data/triage_synthetic_v1.csv

Same seed, same output. triage.eval.build_dataset combines these with the public prompts.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from triage.config import REPO_ROOT

SEED = 2701
VERSION = "synthetic_v1"
SYNTHETIC = REPO_ROOT / "data" / "triage_synthetic_v1.csv"
COLUMNS = [
    "id",
    "use_case",
    "source",
    "source_id",
    "check_type",
    "prompt",
    "answer",
    "override_reason",
    "label",
    "label_reason",
]

# Instruction wording copied from the public items each family mirrors.
SQUAD = (
    "Answer using only the passage below. Return only the shortest answer span. If the passage does not "
    "support an answer, return exactly NOT_IN_DOCUMENT.\n\nPASSAGE:\n{passage}\n\nQUESTION:\n{question}"
)
CUAD = (
    "Read the contract and perform the extraction task. Return only a JSON array of all matching text spans, "
    "copied verbatim from the contract. Return [] if no matching text is present.\n\nCONTRACT:\n{contract}"
    '\n\nTASK:\nHighlight the parts (if any) of this contract related to "{category}" that should be reviewed '
    "by a lawyer. Details: {details}"
)
CX = "CONTRACT TEXT: '{contract}' TASK: {task}"
HUMANEVAL = (
    "Complete the Python function below so that it satisfies its docstring. Return the complete code, "
    "including any imports and helper functions shown, as plain Python without Markdown fences.\n\n{code}"
)
GSM8K = "Solve the problem. Return only the final numeric answer, with no units, symbols or working.\n\nPROBLEM:\n{problem}"
BANKING77 = (
    "Classify the {kind} into exactly one of the intent labels below. Return only the label, exactly as "
    "written.\n\nLABELS:\n{labels}\n\nQUERY:\n{query}"
)

COMPANIES = [
    "Velmora Logistics Ltd",
    "Harrow Fine Foods Inc",
    "Quillstone Analytics Corp",
    "Brightwater Packaging LLC",
    "Corvane Retail Group Ltd",
    "Ashgrove Biotech Inc",
    "Talmont Freight Co",
    "Ninebark Software Ltd",
    "Pellucid Home Goods Inc",
    "Ostrava Coffee Traders LLC",
    "Kestrel Field Services Ltd",
    "Marrowind Paper Co",
    "Thistledown Dairy Cooperative",
    "Saltmarsh Robotics Inc",
    "Calloway & Brandt Distributors Ltd",
    "Ferngate Pharmacy Group Inc",
]
PEOPLE = [
    "Imogen Vasquez-Hale",
    "Tobias Okonkwo",
    "Marguerite Lindqvist",
    "Desmond Achterberg",
    "Priya Rautenbach",
]
PLACES = [
    "Ontario",
    "the State of Delaware",
    "England and Wales",
    "British Columbia",
    "the State of New York",
]
TERRITORIES = [
    "Canada",
    "the Nordic countries",
    "the Benelux region",
    "Western Australia",
    "the Pacific Northwest",
]
CITIES = ["Reykjavik", "Lisbon", "Osaka", "Montreal", "Cape Town"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]


@dataclass
class Item:
    use_case: str
    check_type: str
    prompt: str
    answer: str  # as stored in the CSV: JSON list for code-graded types, raw text for SQL / tests
    label: str
    label_reason: str
    template: str


# --- Document QA: invented policy memos -------------------------------------------------------------------


def _memo(title: str, company: str, sections: list[str]) -> str:
    body = " ".join(f"Section {i}. {s}" for i, s in enumerate(sections, 1))
    return f"{company}: {title}. {body}"


def _qa(memo: str, question: str, answers: list[str], label: str, reason: str, template: str) -> Item:
    return Item(
        "document_qa",
        "exact_match",
        SQUAD.format(passage=memo, question=question),
        json.dumps(answers),
        label,
        reason,
        template,
    )


def document_qa(rng: random.Random) -> list[Item]:
    items = []
    company = rng.choice(COMPANIES)
    adv, hrs, hotel, meal, claim = rng.choice([14, 21]), rng.choice([5, 6]), rng.choice([180, 210]), 65, 30
    hotel_hi, city = hotel + rng.choice([120, 150]), rng.choice(CITIES)
    memo = _memo(
        "Business Travel Policy",
        company,
        [
            f"Flights must be booked through the Wayfarer portal at least {adv} days before departure.",
            f"Economy class is required for flights under {hrs} hours; premium economy may be booked for "
            "longer flights with director approval.",
            f"Hotel costs are reimbursed up to ${hotel} per night in most cities and up to ${hotel_hi} per "
            f"night in {city}.",
            f"Meals are covered by a daily allowance of ${meal}; alcohol is never reimbursable.",
            f"Expense claims must be submitted within {claim} days of returning from the trip.",
        ],
    )
    items.append(
        _qa(
            memo,
            "How many days before departure must flights be booked?",
            [f"{adv} days", f"at least {adv} days", str(adv)],
            "easy",
            "Direct lookup of a single number.",
            "memo_travel_lookup",
        )
    )
    items.append(
        _qa(
            memo,
            f"What is the nightly hotel limit in {city}?",
            [f"${hotel_hi} per night", f"${hotel_hi}", f"up to ${hotel_hi} per night"],
            "hard",
            f"Distractor: the general limit (${hotel}) sits in the same sentence as the {city} limit.",
            "memo_travel_distractor",
        )
    )

    company = rng.choice(COMPANIES)
    t1, t2, t3, t4 = rng.choice([18, 24]), rng.choice([6, 7]), rng.choice([30, 45]), rng.choice([20, 28])
    memo = _memo(
        "Data Retention Standard",
        company,
        [
            f"Customer support tickets are kept for {t1} months after they are closed.",
            f"Payroll records are retained for {t2} years.",
            f"Security camera footage is deleted after {t3} days unless it is linked to an open incident.",
            "Backups of the billing database are encrypted and stored in two separate regions.",
            f"Requests to delete personal data are completed within {t4} days.",
        ],
    )
    items.append(
        _qa(
            memo,
            "How long are payroll records retained?",
            [f"{t2} years"],
            "easy",
            "Direct lookup.",
            "memo_retention_lookup",
        )
    )
    items.append(
        _qa(
            memo,
            "How long are job applicants' CVs retained?",
            ["NOT_IN_DOCUMENT"],
            "hard",
            "Not covered by the memo; several other retention periods invite a guess.",
            "memo_retention_unanswerable",
        )
    )

    company = rng.choice(COMPANIES)
    cert, mins, brk = rng.choice([12, 18]), rng.choice([40, 45]), rng.choice([10, 15])
    stack, stack_low = rng.choice([4, 5]), rng.choice([2, 3])
    memo = _memo(
        "Warehouse Safety Rules",
        company,
        [
            f"Forklift operators must renew their certification every {cert} months.",
            "Visitors must wear high-visibility vests in all aisles and may not enter the cold store.",
            f"Staff may work inside the cold store for at most {mins} minutes at a time, followed by a warm "
            f"break of at least {brk} minutes.",
            f"Pallets may be stacked no higher than {stack} levels, or {stack_low} levels for glass products.",
        ],
    )
    items.append(
        _qa(
            memo,
            "How often must forklift operators renew their certification?",
            [f"every {cert} months", f"{cert} months"],
            "easy",
            "Direct lookup.",
            "memo_safety_lookup",
        )
    )
    items.append(
        _qa(
            memo,
            "How many levels high may pallets of glass products be stacked?",
            [f"{stack_low} levels", str(stack_low), f"{stack_low}"],
            "hard",
            f"Distractor: the general limit ({stack} levels) comes first in the same sentence.",
            "memo_safety_distractor",
        )
    )

    company, owner = rng.choice(COMPANIES), rng.choice(PEOPLE)
    spend, days = rng.choice([250_000, 400_000]), rng.choice([5, 7])
    memo = _memo(
        "Vendor Onboarding Procedure",
        company,
        [
            "New vendors must pass a credit check before their first purchase order is issued.",
            f"Vendors with expected annual spend above ${spend:,} also require a site visit by procurement.",
            "Bank details are verified by calling the vendor on a number taken from its website, never from "
            "the invoice.",
            f"The onboarding review is owned by {owner}, who signs off within {days} business days.",
            "Exceptions to this procedure require written approval from the Chief Financial Officer.",
        ],
    )
    items.append(
        _qa(
            memo,
            "Who owns the onboarding review?",
            [owner],
            "easy",
            "Direct lookup of a name.",
            "memo_vendor_lookup",
        )
    )
    items.append(
        _qa(
            memo,
            "How are vendor bank details verified?",
            [
                "by calling the vendor on a number taken from its website",
                "calling the vendor on a number taken from its website",
                "by calling the vendor on a number taken from its website, never from the invoice",
            ],
            "hard",
            "Long answer span under exact match: trimming or paraphrasing differently fails.",
            "memo_vendor_long_span",
        )
    )

    company = rng.choice(COMPANIES)
    tenure, weeks, weeks_short, pct = rng.choice([6, 12]), rng.choice([16, 20]), rng.choice([8, 10]), 70
    memo = _memo(
        "Parental Leave Policy",
        company,
        [
            f"Employees with at least {tenure} months of service receive {weeks} weeks of fully paid parental "
            "leave.",
            f"Employees with less service receive {weeks_short} weeks of leave at {pct}% of salary.",
            "Leave must start within 12 months of the birth or adoption.",
            "Up to 4 weeks may be taken in separate blocks with manager agreement.",
        ],
    )
    items.append(
        _qa(
            memo,
            "How many weeks of fully paid parental leave do eligible employees receive?",
            [f"{weeks} weeks", str(weeks)],
            "easy",
            "Direct lookup.",
            "memo_leave_lookup",
        )
    )
    items.append(
        _qa(
            memo,
            f"How many weeks of parental leave does an employee with {tenure - 2} months of service receive?",
            [f"{weeks_short} weeks", str(weeks_short), f"{weeks_short} weeks of leave at {pct}% of salary"],
            "hard",
            "Must apply the service condition: the more prominent fully paid figure is wrong here.",
            "memo_leave_condition",
        )
    )

    company = rng.choice(COMPANIES)
    laptop, laptop_eng, phone, desk = (
        rng.choice([4, 5]),
        rng.choice([2, 3]),
        rng.choice([40, 50]),
        rng.choice(["Ext. 4410", "Ext. 7720"]),
    )
    memo = _memo(
        "IT Equipment Policy",
        company,
        [
            f"Laptops are replaced every {laptop} years, or every {laptop_eng} years for engineering staff.",
            f"Staff who use a personal phone for work receive an allowance of ${phone} per month.",
            "Lost or stolen devices must be reported to the service desk within 24 hours.",
            f"The service desk can be reached on {desk} between 8 am and 6 pm on Business Days.",
            "Personal software may not be installed on company devices without written approval.",
        ],
    )
    items.append(
        _qa(
            memo,
            "What is the monthly allowance for staff who use a personal phone for work?",
            [f"${phone} per month", f"${phone}"],
            "easy",
            "Direct lookup.",
            "memo_it_lookup",
        )
    )
    items.append(
        _qa(
            memo,
            "How often are laptops replaced for engineering staff?",
            [f"every {laptop_eng} years", f"{laptop_eng} years"],
            "hard",
            f"Distractor: the general interval ({laptop} years) comes first in the same sentence.",
            "memo_it_distractor",
        )
    )
    return items


# --- Contracts ---------------------------------------------------------------------------------------------

BOILERPLATE = [
    (
        "Confidentiality",
        "Each party shall keep confidential all non-public information received from the other party and "
        "shall use it only to perform this Agreement.",
    ),
    (
        "Force Majeure",
        "Neither party is liable for any delay caused by events beyond its reasonable control, including "
        "fire, flood, strike or failure of public utilities.",
    ),
    (
        "Notices",
        "All notices under this Agreement shall be in writing and delivered by courier or email to the "
        "addresses set out in Schedule 1.",
    ),
    (
        "Entire Agreement",
        "This Agreement is the entire agreement between the parties and supersedes all prior discussions "
        "relating to its subject matter.",
    ),
    (
        "Severability",
        "If any provision of this Agreement is held invalid, the remaining provisions remain in full force.",
    ),
    ("Counterparts", "This Agreement may be signed in counterparts, each of which is an original."),
    (
        "Warranties",
        "{A} warrants that the Products will conform to the Specifications for twelve months from delivery.",
    ),
    (
        "Delivery",
        "{A} shall deliver the Products to the facility named in each purchase order, freight prepaid.",
    ),
    (
        "Intellectual Property",
        "Each party retains all intellectual property it owned before the Effective Date; nothing in this "
        "Agreement transfers ownership of it.",
    ),
    ("Records", "{B} shall keep accurate records of all Products received and their storage conditions."),
    (
        "Independent Contractors",
        "The parties are independent contractors, and nothing in this Agreement creates a partnership or "
        "agency.",
    ),
    ("Waiver", "A failure to enforce any provision is not a waiver of the right to enforce it later."),
    (
        "Product Recalls",
        "{A} shall notify {B} within two business days of any recall affecting the Products.",
    ),
    ("Headings", "Headings are for convenience only and do not affect interpretation."),
]


def _clauses(rng: random.Random, a: str, b: str) -> tuple[dict[str, tuple[str, str]], dict]:
    """Clause type -> (heading, text), and the numbers drawn for this contract."""
    n = {k: rng.choice(v) for k, v in {
        "assign_notice": [10, 15], "term": [60, 90, 120], "renew_m": [12, 24], "renew_notice": [45, 75],
        "noncompete_m": [12, 18], "nonsolicit_m": [6, 12], "cap_m": [6, 12], "audit": [20, 30],
        "insure": [1_000_000, 2_000_000, 5_000_000], "pay": [30, 45, 60], "interest": [1.25, 1.5, 2.0],
    }.items()}  # fmt: skip
    territory, place = rng.choice(TERRITORIES), rng.choice(PLACES)
    return {
        "assign_restrict": (
            "Assignment",
            f"{b} shall not assign or transfer this Agreement or any of its rights hereunder, in whole or in "
            f"part, without the prior written consent of {a}, which consent shall not be unreasonably withheld.",
        ),
        "assign_free": (
            "Assignment by " + a.split()[0],
            f"{a} may assign this Agreement to any Affiliate or to a successor of its business without the "
            f"consent of {b}, provided that {a} gives {b} written notice within {n['assign_notice']} days after "
            "such assignment.",
        ),
        "governing_law": (
            "Governing Law",
            f"This Agreement shall be governed by and construed in accordance with the laws of {place}, without "
            "regard to its conflict of laws principles.",
        ),
        "termination_convenience": (
            "Termination for Convenience",
            f"{b} may terminate this Agreement for convenience at any time upon {n['term']} days' prior "
            f"written notice to {a}.",
        ),
        "renewal": (
            "Renewal",
            f"Upon expiry of the Initial Term, this Agreement shall renew automatically for successive periods "
            f'of {n["renew_m"]} months each (each a "Renewal Term").',
        ),
        "renewal_notice": (
            "Non-Renewal",
            f"Either party may prevent a renewal by giving the other party written notice of non-renewal at "
            f"least {n['renew_notice']} days before the end of the then-current term.",
        ),
        "non_compete": (
            "Non-Competition",
            f"During the Term and for {n['noncompete_m']} months thereafter, {b} shall not, directly or "
            f"indirectly, manufacture or sell any product that competes with the Products within {territory}.",
        ),
        "non_solicit": (
            "Non-Solicitation",
            f"During the Term and for {n['nonsolicit_m']} months thereafter, neither party shall solicit for "
            "employment any employee of the other party who was involved in the performance of this Agreement.",
        ),
        "liability_cap": (
            "Limitation of Liability",
            "Except for breaches of confidentiality, each party's total liability arising out of or relating "
            f"to this Agreement shall not exceed the fees paid by {b} in the {n['cap_m']} months preceding the "
            "event giving rise to the claim.",
        ),
        "audit": (
            "Audit",
            f"{a} may, upon {n['audit']} days' written notice and no more than once per calendar year, audit "
            f"the books and records of {b} relating to the Products.",
        ),
        "insurance": (
            "Insurance",
            f"{b} shall maintain commercial general liability insurance with limits of not less than "
            f"${n['insure']:,} per occurrence throughout the Term.",
        ),
        "payment": (
            "Payment",
            f"{b} shall pay each undisputed invoice within {n['pay']} days of the invoice date. Late amounts "
            f"bear interest at {n['interest']}% per month.",
        ),
    }, n


# Filler that makes the CUAD-style contracts as long as real ones (CUAD median ~24K characters). None of it
# belongs to a category we ask about: no inspections (Audit Rights), no subcontracting limits
# (Anti-Assignment), no dispute venue (Governing Law), no exclusivity (Non-Compete), no damages exclusions
# (Cap On Liability), no other termination rights.
OPERATIONS = [
    (
        "Forecasts",
        "{B} shall provide {A} with a rolling non-binding forecast of its requirements for the Products, updated at the start of each calendar month. Forecasts are for planning purposes only and do not create any obligation to purchase.",
    ),
    (
        "Purchase Orders",
        "{B} shall order Products by issuing written purchase orders that state the Products, quantities, requested delivery dates and the delivery location. Each purchase order is governed by this Agreement, and any additional or different terms in a purchase order or acknowledgement have no effect.",
    ),
    (
        "Order Acknowledgement",
        "{A} shall acknowledge each purchase order in writing, confirming the delivery date or proposing an alternative date that it can reasonably meet. A purchase order is accepted when acknowledged.",
    ),
    (
        "Packaging and Labelling",
        "{A} shall pack the Products in accordance with the Packaging Standards in Schedule C, so that they reach the delivery location undamaged. Each carton shall show the product code, batch number, quantity and the purchase order number.",
    ),
    (
        "Acceptance",
        "{B} may reject any delivery of Products that does not conform to the Specifications by notifying {A} in writing with reasonable details of the non-conformity. Rejected Products shall be collected by {A} at its own cost and replaced or credited at the option of {B}.",
    ),
    (
        "Title and Risk",
        "Title to and risk in the Products pass to {B} when the Products are unloaded at the delivery location, except that risk in rejected Products returns to {A} on notice of rejection.",
    ),
    (
        "Invoicing",
        "{A} shall invoice {B} for each delivery after it is made. Each invoice shall state the purchase order number, the Products delivered, the quantities and the applicable prices from Schedule B.",
    ),
    (
        "Taxes",
        "All prices exclude sales, use and value added taxes, which {B} shall pay in addition at the applicable rate on receipt of a valid tax invoice. Each party is responsible for taxes on its own income.",
    ),
    (
        "Price Review",
        "The parties shall meet once in each Contract Year to review the prices in Schedule B in good faith, taking into account changes in the cost of raw materials, energy and transport.",
    ),
    (
        "Specifications",
        "{A} shall manufacture the Products in accordance with the Specifications and all applicable laws. {A} shall not change the Specifications, the materials used or the place of manufacture without the prior written approval of {B}.",
    ),
    (
        "Quality Management",
        "{A} shall maintain a quality management system that meets the requirements in Schedule D and shall keep certificates of analysis for each batch of Products for the period stated there.",
    ),
    (
        "Compliance with Laws",
        "Each party shall comply with all laws and regulations that apply to its performance of this Agreement, including those relating to product safety, labelling, export control and the environment.",
    ),
    (
        "Anti-Bribery",
        "Neither party shall offer, give or accept any bribe or improper payment in connection with this Agreement, and each party shall maintain policies and procedures designed to prevent bribery by its personnel.",
    ),
    (
        "Data Protection",
        "Where a party processes personal data on behalf of the other party, it shall do so only on documented instructions, keep the data secure and notify the other party without undue delay of any personal data breach.",
    ),
    (
        "Business Continuity",
        "{A} shall maintain a written business continuity plan for the manufacture and supply of the Products and shall provide a summary of it to {B} on request.",
    ),
    (
        "Account Management",
        "Each party shall appoint an account manager who is responsible for the day-to-day relationship under this Agreement. The account managers shall meet at least once in each calendar quarter to review performance against Schedule E.",
    ),
    (
        "Changes",
        "Either party may propose changes to the Products, the delivery arrangements or Schedule C. No change is effective until it is agreed in writing and signed by both parties.",
    ),
    (
        "Publicity",
        "Neither party shall issue any press release or public statement about this Agreement without the prior written approval of the other party.",
    ),
    (
        "Costs",
        "Except as otherwise stated in this Agreement, each party shall pay its own costs of negotiating, preparing and performing this Agreement.",
    ),
    (
        "Further Assurance",
        "Each party shall sign such further documents and do such further acts as the other party may reasonably require to give full effect to this Agreement.",
    ),
    (
        "Third Party Rights",
        "A person who is not a party to this Agreement has no right to enforce any of its terms.",
    ),
    (
        "Variation",
        "No variation of this Agreement is effective unless it is in writing and signed by an authorised representative of each party.",
    ),
]

DEFINED_TERMS = [
    (
        "Affiliate",
        "any entity that directly or indirectly controls, is controlled by, or is under common control with a party, where control means owning more than half of the voting shares",
    ),
    (
        "Business Day",
        "a day other than a Saturday, Sunday or public holiday in the place where the relevant obligation is to be performed",
    ),
    ("Contract Year", "each period of twelve months starting on the Effective Date or an anniversary of it"),
    ("Delivery Location", "the warehouse or facility of {B} named in the relevant purchase order"),
    ("Initial Term", "the period starting on the Effective Date and ending on the third anniversary of it"),
    (
        "Packaging Standards",
        "the packaging, palletisation and labelling requirements set out in Schedule C, as updated from time to time by agreement",
    ),
    (
        "Products",
        "the products listed in Schedule A, as that schedule is updated from time to time by agreement",
    ),
    (
        "Specifications",
        "the specifications for each Product set out or referred to in Schedule A, including the approved formulation, materials and tolerances",
    ),
    ("Service Levels", "the delivery and quality performance levels set out in Schedule E"),
    ("Term", "the Initial Term together with any Renewal Term"),
    ("Batch", "a quantity of a Product manufactured in a single production run under the same conditions"),
    (
        "Certificate of Analysis",
        "a document issued by {A} for a Batch confirming that it was tested and meets the Specifications",
    ),
    (
        "Confidential Information",
        "all information of a party that is marked as confidential or that a reasonable person would understand to be confidential",
    ),
    (
        "Intellectual Property",
        "patents, trade marks, designs, copyright, know-how and all similar rights anywhere in the world",
    ),
    (
        "Lead Time",
        "the period between acceptance of a purchase order and the requested delivery date, as set out for each Product in Schedule A",
    ),
    (
        "Non-Conforming Product",
        "a Product that does not meet the Specifications or that is damaged on arrival at the Delivery Location",
    ),
    ("Personnel", "the employees, agents and contractors of a party"),
    ("Purchase Order", "a written order for Products issued by {B} in accordance with this Agreement"),
]

ADJECTIVES = [
    "Alpine",
    "Coastal",
    "Golden",
    "Harvest",
    "Meadow",
    "Northern",
    "Prairie",
    "Riverbend",
    "Summit",
    "Willow",
]
NOUNS = [
    "Granola",
    "Oat Bar",
    "Rice Crisp",
    "Seed Mix",
    "Muesli",
    "Trail Mix",
    "Corn Flake",
    "Bran Flake",
    "Nut Cluster",
    "Rye Crispbread",
]


def _schedules(rng: random.Random, b: str) -> str:
    products, specs = [], []
    for i in range(rng.randint(26, 34)):
        code = f"PRD-{rng.randint(10000, 99999)}"
        name = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {rng.choice([250, 375, 500, 750])}g"
        products.append(
            f"{code} | {name} | case of {rng.choice([6, 8, 12, 24])} | "
            f"lead time {rng.choice([2, 3, 4, 6])} weeks | shelf life {rng.choice([6, 9, 12, 18])} months"
        )
        if i < 24:
            specs.append(
                f"{code} {name}. Moisture no more than {rng.choice([3, 4, 5, 6])}.{rng.randint(0, 9)}%. Net "
                f"weight tolerance plus or minus {rng.choice([1, 2, 3])}%. Pieces shall be whole and evenly "
                f"coloured, with broken pieces making up no more than {rng.choice([5, 8, 10])}% of a pack by "
                f"weight. Allergen statement: {rng.choice(['contains oats and wheat', 'contains tree nuts', 'contains sesame', 'contains milk and soy'])}; "
                f"made in a facility that also handles {rng.choice(['peanuts', 'sesame', 'mustard', 'lupin'])}. "
                f"Store below {rng.choice([20, 22, 25])} degrees Celsius in dry conditions, away from direct "
                "sunlight. Each pack shall carry a best-before date and Batch number printed in a legible, "
                "indelible format."
            )
    prices = [
        f"{lo:,} to {lo * 4 - 1:,} cases per quarter: ${rng.randint(18, 40)}.{rng.randint(0, 99):02d} per case"
        for lo in (1, 2_000, 8_000, 32_000)
    ]
    sites = [f"{rng.choice(CITIES)} Distribution Centre, Dock {rng.randint(1, 12)}" for _ in range(4)]
    return "\n\n".join(
        [
            "SCHEDULE A - PRODUCTS AND SPECIFICATIONS\nCode | Product | Pack | Lead time | Shelf life\n"
            + "\n".join(products),
            "SCHEDULE A1 - PRODUCT SPECIFICATIONS\n" + "\n\n".join(specs),
            "SCHEDULE B - PRICES\n" + "\n".join(prices) + "\nPrices are in US dollars per case delivered.",
            "SCHEDULE C - PACKAGING STANDARDS\nCartons shall be double-walled corrugated board, stacked on "
            "standard 1200 x 1000 mm pallets to a height of no more than 1.8 metres, shrink-wrapped and labelled "
            "on two adjacent sides. Each pallet shall carry a single Product and Batch unless agreed otherwise. "
            "Labels shall be printed in black on white, in English and French, and shall carry a scannable "
            "barcode with the product code and Batch number.",
            "SCHEDULE D - QUALITY REQUIREMENTS\nThe quality management system shall be certified to a recognised "
            "food safety standard. Each Batch shall be tested for moisture, foreign matter and allergen "
            "cross-contact before release. Certificates of Analysis shall be kept for two years after the end of "
            "the shelf life of the Batch and supplied with each delivery. Retained samples of each Batch shall be "
            "stored under controlled conditions until the end of its shelf life.",
            "SCHEDULE E - SERVICE LEVELS\nOn-time delivery: at least 97% of purchase order lines delivered on "
            "the requested date, measured monthly.\nIn-full delivery: at least 98% of ordered cases delivered, "
            "measured monthly.\nQuality: no more than 0.5% of delivered cases rejected as Non-Conforming Product, "
            "measured quarterly.\nThe account managers shall review performance against these levels at each "
            "quarterly meeting and agree a written improvement plan for any level that is missed.",
            f"SCHEDULE F - DELIVERY LOCATIONS OF {b.upper()}\n" + "\n".join(sites),
        ]
    )


def _contract(
    rng: random.Random,
    picks: list[str],
    n_boiler: int,
    a_role: str = "Supplier",
    b_role: str = "Customer",
    long: bool = False,
) -> tuple[str, dict, dict, dict, str]:
    """A supply agreement. long=True adds definitions, operational clauses and schedules (~20K characters)."""
    a, b = rng.sample(COMPANIES, 2)
    clauses, nums = _clauses(rng, a, b)
    boiler = [(h, t.format(A=a, B=b)) for h, t in rng.sample(BOILERPLATE, n_boiler)]
    body = [clauses[k] for k in picks] + boiler
    if long:
        body += [(h, t.format(A=a, B=b)) for h, t in rng.sample(OPERATIONS, rng.randint(16, len(OPERATIONS)))]
    rng.shuffle(body)
    day, month, year = rng.randint(2, 27), rng.choice(MONTHS), rng.choice([2025, 2026])
    header = (
        f'SUPPLY AGREEMENT\n\nThis Supply Agreement (the "Agreement") is entered into on {month} {day}, {year} '
        f'(the "Effective Date") between {a} ("{a_role}") and {b} ("{b_role}").'
    )
    sections = [f"{i}. {h}. {t}" for i, (h, t) in enumerate(body, 2)]
    if long:
        terms = sorted(DEFINED_TERMS)
        sections.insert(
            0,
            "1. Definitions. In this Agreement the following terms have these meanings:\n"
            + "\n".join(f'"{term}" means {meaning.format(A=a, B=b)};' for term, meaning in terms),
        )
    else:
        sections = [f"{i}. {h}. {t}" for i, (h, t) in enumerate(body, 1)]
    text = header + "\n\n" + "\n\n".join(sections)
    text += f"\n\nIN WITNESS WHEREOF the parties have signed this Agreement on the Effective Date.\n{a}\n{b}"
    if long:
        text += "\n\n" + _schedules(rng, b)
    meta = {"a": a, "b": b, "date": (year, MONTHS.index(month) + 1, day)}
    return text, clauses, nums, meta, header


def _span_item(
    contract: str, category: str, details: str, gold: list[str], label: str, reason: str, t: str
) -> Item:
    for span in gold:
        assert span in contract, f"{t}: gold span not in contract"
    prompt = CUAD.format(contract=contract, category=category, details=details)
    return Item("contract_extraction", "span_set", prompt, json.dumps(gold), label, reason, t)


def contracts(rng: random.Random) -> list[Item]:
    items = []
    text, c, *_ = _contract(rng, ["governing_law", "payment", "audit", "insurance"], 5, long=True)
    items.append(
        _span_item(
            text,
            "Governing Law",
            "Which state/country's law governs the interpretation of the contract?",
            [c["governing_law"][1]],
            "easy",
            "One clearly headed clause.",
            "contract_span_governing_law",
        )
    )
    text, c, *_ = _contract(rng, ["termination_convenience", "renewal", "payment"], 5, long=True)
    items.append(
        _span_item(
            text,
            "Termination For Convenience",
            "Can a party terminate this contract without cause (solely by giving a notice and allowing a "
            "waiting period to expire)?",
            [c["termination_convenience"][1]],
            "easy",
            "One clearly headed clause.",
            "contract_span_termination",
        )
    )
    text, c, *_ = _contract(rng, ["audit", "payment", "governing_law"], 6, long=True)
    items.append(
        _span_item(
            text,
            "Audit Rights",
            "Does a party have the right to audit the books, records, or physical locations of the "
            "counterparty to ensure compliance with the contract?",
            [c["audit"][1]],
            "easy",
            "One clearly headed clause.",
            "contract_span_audit",
        )
    )
    text, c, *_ = _contract(rng, ["assign_restrict", "assign_free", "governing_law", "payment"], 6, long=True)
    items.append(
        _span_item(
            text,
            "Anti-Assignment",
            "Is consent or notice required of a party if the contract is assigned to a third party?",
            [c["assign_restrict"][1], c["assign_free"][1]],
            "hard",
            "Two assignment clauses with mirrored parties: one needs consent, the other only notice; both must "
            "be returned, and they differ by who may assign.",
            "contract_span_assignment_mirrored",
        )
    )
    text, c, *_ = _contract(
        rng, ["non_solicit", "governing_law", "termination_convenience", "payment"], 7, long=True
    )
    items.append(
        _span_item(
            text,
            "Non-Compete",
            "Is there a restriction on the ability of a party to compete with the counterparty or operate in "
            "a certain geography or business or technology sector?",
            [],
            "hard",
            "No non-compete; a non-solicitation clause with the same 'During the Term and for N months "
            "thereafter' shape invites a false positive. Answer is [].",
            "contract_span_noncompete_absent",
        )
    )
    picks = ["liability_cap", "insurance", "payment", "audit", "governing_law", "renewal", "renewal_notice"]
    text, c, *_ = _contract(rng, picks, 14, long=True)
    items.append(
        _span_item(
            text,
            "Cap On Liability",
            "Does the contract include a cap on liability upon the breach of a party's obligation?",
            [c["liability_cap"][1]],
            "hard",
            "Every boilerplate clause included; a dollar-limit insurance clause is a distractor.",
            "contract_span_liability_long",
        )
    )

    text, c, *_ = _contract(rng, ["renewal", "payment", "governing_law"], 6, long=True)
    items.append(
        _span_item(
            text,
            "Renewal Term",
            "What is the renewal term after the initial term expires? This includes automatic extensions and "
            "unilateral extensions with prior notice.",
            [c["renewal"][1]],
            "easy",
            "One clearly headed clause.",
            "contract_span_renewal",
        )
    )
    text, c, *_ = _contract(
        rng, ["renewal", "renewal_notice", "termination_convenience", "payment"], 8, long=True
    )
    items.append(
        _span_item(
            text,
            "Notice Period To Terminate Renewal",
            "What is the notice period required to terminate renewal?",
            [c["renewal_notice"][1]],
            "hard",
            "The renewal clause itself and a termination-for-convenience notice period are close distractors; "
            "only the non-renewal notice clause answers it.",
            "contract_span_renewal_notice",
        )
    )

    def cx(contract: str, task: str, gold: dict, label: str, reason: str, t: str) -> Item:
        prompt = CX.format(contract=" ".join(contract.split()), task=task)
        return Item("contract_extraction", "json_match", prompt, json.dumps([gold]), label, reason, t)

    text, c, nums, meta, _ = _contract(rng, ["payment", "governing_law"], 2, "Supplier", "Buyer")
    y, m, d = meta["date"]
    items.append(
        cx(
            text,
            "Extract the supplier name, buyer name and effective date as JSON with keys supplier, buyer, "
            "effective_date (ISO format).",
            {"supplier": meta["a"], "buyer": meta["b"], "effective_date": f"{y}-{m:02d}-{d:02d}"},
            "easy",
            "Fields are in the first sentence.",
            "contract_json_parties",
        )
    )
    text, c, nums, meta, _ = _contract(rng, ["payment", "audit"], 2)
    items.append(
        cx(
            text,
            "Extract the payment terms as JSON with keys net_days and late_interest_monthly_pct (numbers).",
            {"net_days": nums["pay"], "late_interest_monthly_pct": nums["interest"]},
            "easy",
            "Both numbers are in one clause.",
            "contract_json_payment",
        )
    )
    text, c, nums, meta, _ = _contract(
        rng, ["termination_convenience", "renewal", "renewal_notice", "payment", "audit"], 3
    )
    items.append(
        cx(
            text,
            "Extract the notice periods as JSON with keys termination_notice_days (notice to terminate for "
            "convenience) and renewal_notice_days (notice to prevent a renewal), as numbers.",
            {"termination_notice_days": nums["term"], "renewal_notice_days": nums["renew_notice"]},
            "hard",
            "Four different day counts (termination, non-renewal, payment, audit); each key needs the right one.",
            "contract_json_notice_periods",
        )
    )
    text, c, nums, meta, _ = _contract(rng, ["liability_cap", "insurance", "payment", "renewal"], 3)
    items.append(
        cx(
            text,
            "Extract as JSON with keys liability_cap_months (how many months of fees the liability cap is "
            "based on) and insurance_minimum_usd (minimum insurance limit per occurrence), as numbers.",
            {"liability_cap_months": nums["cap_m"], "insurance_minimum_usd": nums["insure"]},
            "hard",
            "Cap is expressed in months of fees, not dollars; renewal months are a distractor; the insurance "
            "figure must be returned as a plain number.",
            "contract_json_cap_insurance",
        )
    )
    return items


# --- Coding: invented function specs ----------------------------------------------------------------------


@dataclass
class CodeTask:
    name: str
    label: str
    reason: str
    stub: str  # signature + docstring shown to the model
    tests: str  # body of check(candidate)
    reference: str  # our solution; never shown, only used to verify the tests


CODE_TASKS = [
    CodeTask(
        "loyalty_points",
        "easy",
        "One integer rule.",
        '''def loyalty_points(amount_cents):
    """Return the loyalty points earned on an order of amount_cents cents.
    Customers earn 3 points for every full $10 spent.

    loyalty_points(2500) == 6
    loyalty_points(999) == 0
    """
''',
        """    assert candidate(0) == 0
    assert candidate(999) == 0
    assert candidate(1000) == 3
    assert candidate(2500) == 6
    assert candidate(19999) == 57
""",
        """def loyalty_points(amount_cents):
    return amount_cents // 1000 * 3
""",
    ),
    CodeTask(
        "ticket_id",
        "easy",
        "String formatting with zero padding.",
        '''def ticket_id(region, number):
    """Return a support ticket ID: the region code in upper case, a dash, then the number
    zero-padded to 6 digits. Numbers longer than 6 digits are kept as they are.

    ticket_id("emea", 42) == "EMEA-000042"
    """
''',
        """    assert candidate("emea", 42) == "EMEA-000042"
    assert candidate("Apac", 123456) == "APAC-123456"
    assert candidate("na", 0) == "NA-000000"
    assert candidate("na", 1234567) == "NA-1234567"
""",
        """def ticket_id(region, number):
    return f"{region.upper()}-{number:06d}"
""",
    ),
    CodeTask(
        "count_vowel_runs",
        "easy",
        "Simple scan.",
        '''def count_vowel_runs(word):
    """Return how many separate runs of consecutive vowels (a, e, i, o, u; case-insensitive)
    the word contains.

    count_vowel_runs("banana") == 3
    count_vowel_runs("rhythm") == 0
    """
''',
        """    assert candidate("banana") == 3
    assert candidate("rhythm") == 0
    assert candidate("Aeon") == 1
    assert candidate("cooperate") == 4
    assert candidate("") == 0
""",
        """import re

def count_vowel_runs(word):
    return len(re.findall("[aeiou]+", word.lower()))
""",
    ),
    CodeTask(
        "split_batches",
        "easy",
        "Small arithmetic rule.",
        '''def split_batches(total, max_batch):
    """Split total items into the fewest batches of at most max_batch items each, with batch
    sizes as equal as possible. Return the sizes, larger batches first.

    split_batches(10, 4) == [4, 3, 3]
    split_batches(0, 5) == []
    """
''',
        """    assert candidate(10, 4) == [4, 3, 3]
    assert candidate(8, 4) == [4, 4]
    assert candidate(0, 5) == []
    assert candidate(7, 10) == [7]
    assert candidate(11, 3) == [3, 3, 3, 2]
""",
        """def split_batches(total, max_batch):
    if total == 0:
        return []
    n = -(-total // max_batch)
    q, r = divmod(total, n)
    return [q + 1] * r + [q] * (n - r)
""",
    ),
    CodeTask(
        "shelf_label",
        "easy",
        "String formatting.",
        '''def shelf_label(name, price_cents):
    """Return a shelf label "NAME | $D.CC": the name trimmed and in upper case, then the price
    in dollars with exactly two decimals and no thousands separator.

    shelf_label(" oat milk ", 349) == "OAT MILK | $3.49"
    """
''',
        """    assert candidate(" oat milk ", 349) == "OAT MILK | $3.49"
    assert candidate("tea", 5) == "TEA | $0.05"
    assert candidate("Big Box", 120000) == "BIG BOX | $1200.00"
""",
        """def shelf_label(name, price_cents):
    return f"{name.strip().upper()} | ${price_cents // 100}.{price_cents % 100:02d}"
""",
    ),
    CodeTask(
        "loyalty_points_v2",
        "hard",
        "Five interacting rules; order of operations matters (minimum order, doubling, tier bonus, cap).",
        '''def loyalty_points_v2(amount_cents, weekday, tier):
    """Return the loyalty points for an order.
    - Orders under $5.00 earn nothing.
    - Otherwise customers earn 3 points for every full $10 spent.
    - On Tuesdays (weekday == "Tue") the points are doubled.
    - Gold members (tier == "gold") then get a bonus of 25% of their points, rounded down.
    - No order earns more than 500 points.

    loyalty_points_v2(2500, "Tue", "gold") == 15
    """
''',
        """    assert candidate(499, "Tue", "gold") == 0
    assert candidate(900, "Tue", "gold") == 0
    assert candidate(2500, "Mon", "basic") == 6
    assert candidate(2500, "Tue", "basic") == 12
    assert candidate(2500, "Tue", "gold") == 15
    assert candidate(4000, "Mon", "gold") == 15
    assert candidate(300000, "Tue", "gold") == 500
""",
        """def loyalty_points_v2(amount_cents, weekday, tier):
    if amount_cents < 500:
        return 0
    points = amount_cents // 1000 * 3
    if weekday == "Tue":
        points *= 2
    if tier == "gold":
        points += points // 4
    return min(points, 500)
""",
    ),
    CodeTask(
        "parse_record",
        "hard",
        "Small parser with quoting, escapes, bare keys and whitespace rules.",
        '''def parse_record(line):
    """Parse a record line into a dict.
    - Fields are separated by ";". Empty fields are ignored.
    - A field is either key=value or a bare key, which maps to True.
    - Keys and unquoted values have surrounding whitespace removed.
    - A value may be wrapped in double quotes: inside the quotes ";" is literal, \\\\" is an escaped
      quote, and the text is kept exactly (no trimming). The quotes themselves are removed.
    - If a key repeats, the later value wins.

    parse_record('name=Ana; city = Lyon') == {"name": "Ana", "city": "Lyon"}
    parse_record('note="a;b"; urgent') == {"note": "a;b", "urgent": True}
    """
''',
        """    assert candidate('name=Ana; city = Lyon') == {"name": "Ana", "city": "Lyon"}
    assert candidate('note="a;b"; urgent') == {"note": "a;b", "urgent": True}
    assert candidate('q="say \\\\"hi\\\\""') == {"q": 'say "hi"'}
    assert candidate('a=1;;b=2;') == {"a": "1", "b": "2"}
    assert candidate('x=1; x=2') == {"x": "2"}
    assert candidate('pad="  keep  "') == {"pad": "  keep  "}
    assert candidate('') == {}
""",
        """def parse_record(line):
    fields, buf, in_quotes, i = [], [], False, 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == "\\\\" and i + 1 < len(line) and line[i + 1] == '"':
                buf.append('\\\\"')
                i += 2
                continue
            if ch == '"':
                in_quotes = False
            buf.append(ch)
        elif ch == '"':
            in_quotes = True
            buf.append(ch)
        elif ch == ";":
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    fields.append("".join(buf))
    out = {}
    for field in fields:
        if not field.strip():
            continue
        key, sep, value = field.partition("=")
        key = key.strip()
        if not sep:
            out[key] = True
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\\\"', '"')
        out[key] = value
    return out
""",
    ),
    CodeTask(
        "next_service_day",
        "hard",
        "Two competing limits, day indexing and the never-reached case.",
        '''def next_service_day(last_service_day, daily_hours):
    """A machine needs service after 250 operating hours or 90 days since its last service,
    whichever comes first. last_service_day is the day number of the last service;
    daily_hours[i] is the hours run on day last_service_day + 1 + i.
    Return the day number on which service becomes due: the first day on which the hours since
    the last service reach 250, or last_service_day + 90 if that comes first or the hours never
    reach 250.

    next_service_day(100, [10] * 30) == 125
    """
''',
        """    assert candidate(100, [10] * 30) == 125
    assert candidate(0, [1] * 200) == 90
    assert candidate(10, []) == 100
    assert candidate(5, [250]) == 6
    assert candidate(0, [100, 100, 49, 1]) == 4
    assert candidate(0, [0] * 89 + [300]) == 90
    assert candidate(0, [0] * 90 + [300]) == 90
""",
        """def next_service_day(last_service_day, daily_hours):
    total = 0
    for i, hours in enumerate(daily_hours):
        day = last_service_day + 1 + i
        if day > last_service_day + 90:
            break
        total += hours
        if total >= 250:
            return day
    return last_service_day + 90
""",
    ),
    CodeTask(
        "merge_shifts",
        "hard",
        "Interval merging on clock strings, touching intervals, and a minimum length applied after merging.",
        '''def merge_shifts(shifts):
    """Each shift is a (start, end) tuple of "HH:MM" strings within one day, with end after start.
    Merge shifts that overlap or touch. Drop merged shifts shorter than 15 minutes.
    Return the merged shifts as (start, end) tuples sorted by start.

    merge_shifts([("09:00", "12:00"), ("11:30", "13:00")]) == [("09:00", "13:00")]
    """
''',
        """    assert candidate([("09:00", "12:00"), ("11:30", "13:00")]) == [("09:00", "13:00")]
    assert candidate([("08:00", "09:00"), ("09:00", "10:00")]) == [("08:00", "10:00")]
    assert candidate([("14:00", "15:00"), ("07:05", "07:30")]) == [("07:05", "07:30"), ("14:00", "15:00")]
    assert candidate([("10:00", "10:10")]) == []
    assert candidate([("10:00", "10:10"), ("10:05", "10:30")]) == [("10:00", "10:30")]
    assert candidate([]) == []
""",
        """def merge_shifts(shifts):
    def minutes(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    merged = []
    for a, b in sorted((minutes(a), minutes(b)) for a, b in shifts):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(f"{a // 60:02d}:{a % 60:02d}", f"{b // 60:02d}:{b % 60:02d}") for a, b in merged if b - a >= 15]
""",
    ),
    CodeTask(
        "invoice_total",
        "hard",
        "Tax exemption, a capped discount on part of the order, tax on the discounted amount, half-up rounding.",
        '''def invoice_total(lines, coupon=None):
    """Return an invoice total in cents.
    lines is a list of (quantity, unit_price_cents, category) tuples.
    - Tax is 13% on every category except "food", which is not taxed.
    - coupon "SAVE10" takes 10% off the pre-tax subtotal of the non-food lines, at most 2000 cents;
      tax is charged on the discounted amount. Any other coupon raises ValueError.
    - Round once, at the end, to the nearest cent; halves round up.

    invoice_total([(1, 1000, "tools"), (1, 500, "food")], "SAVE10") == 1517
    """
''',
        """    assert candidate([(2, 500, "food")]) == 1000
    assert candidate([(1, 1000, "tools")]) == 1130
    assert candidate([(1, 1000, "tools"), (1, 500, "food")], "SAVE10") == 1517
    assert candidate([(1, 30000, "tools")], "SAVE10") == 31640
    assert candidate([(3, 333, "tools")]) == 1129
    assert candidate([(1, 5, "tools")]) == 6
    assert candidate([]) == 0
    try:
        candidate([(1, 100, "tools")], "BOGUS")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown coupon must raise ValueError")
""",
        """from decimal import ROUND_HALF_UP, Decimal

def invoice_total(lines, coupon=None):
    food = sum(q * u for q, u, c in lines if c == "food")
    other = sum(q * u for q, u, c in lines if c != "food")
    discount = Decimal(0)
    if coupon is not None:
        if coupon != "SAVE10":
            raise ValueError(coupon)
        discount = min(Decimal(other) / 10, Decimal(2000))
    total = Decimal(food) + (Decimal(other) - discount) * Decimal("1.13")
    return int(total.quantize(Decimal(1), rounding=ROUND_HALF_UP))
""",
    ),
    CodeTask(
        "initials",
        "easy",
        "Simple string handling.",
        """def initials(full_name):
    \"\"\"Return the initials of a name: the first letter of each word in upper case, each followed
    by a full stop. Words are separated by spaces; extra spaces are ignored.

    initials("ana maria lopez") == "A.M.L."
    \"\"\"
""",
        """    assert candidate("ana maria lopez") == "A.M.L."
    assert candidate("  Tobias   Okonkwo ") == "T.O."
    assert candidate("x") == "X."
    assert candidate("") == ""
""",
        """def initials(full_name):
    return "".join(word[0].upper() + "." for word in full_name.split())
""",
    ),
    CodeTask(
        "count_by_week",
        "hard",
        "ISO week numbering: dates near New Year belong to the neighbouring year's week.",
        """def count_by_week(dates):
    \"\"\"dates is a list of "YYYY-MM-DD" strings. Return a dict mapping each ISO week, written
    "YYYY-Www" with the ISO week-numbering year and a two-digit week, to the number of dates in it.

    count_by_week(["2026-06-15", "2026-06-16"]) == {"2026-W25": 2}
    \"\"\"
""",
        """    assert candidate(["2026-06-15", "2026-06-16"]) == {"2026-W25": 2}
    assert candidate(["2026-01-01"]) == {"2026-W01": 1}
    assert candidate(["2025-12-29", "2026-01-01"]) == {"2026-W01": 2}
    assert candidate(["2026-12-31", "2027-01-01", "2027-01-04"]) == {"2026-W53": 2, "2027-W01": 1}
    assert candidate([]) == {}
""",
        """import datetime

def count_by_week(dates):
    counts = {}
    for d in dates:
        year, week, _ = datetime.date.fromisoformat(d).isocalendar()
        key = f"{year}-W{week:02d}"
        counts[key] = counts.get(key, 0) + 1
    return counts
""",
    ),
]


def code_tests(task: CodeTask) -> str:
    return f"def check(candidate):\n{task.tests}\ncheck({task.name})"


def coding(rng: random.Random) -> list[Item]:
    return [
        Item(
            "coding",
            "unit_tests",
            HUMANEVAL.format(code=t.stub),
            code_tests(t),
            t.label,
            t.reason,
            f"code_{t.name}",
        )
        for t in CODE_TASKS
    ]


# --- Misc: arithmetic and an invented intent set ----------------------------------------------------------

HELPDESK_LABELS = [
    "roster_swap_request",
    "badge_access_issue",
    "payslip_discrepancy",
    "overtime_approval",
    "leave_balance_query",
    "uniform_order",
    "vehicle_damage_report",
    "fuel_card_declined",
    "route_change_request",
    "training_signup",
    "timesheet_correction",
    "equipment_return",
]


def misc(rng: random.Random) -> list[Item]:
    items = []

    def math(problem: str, answer: float, label: str, reason: str, t: str) -> Item:
        return Item(
            "misc_math", "numeric", GSM8K.format(problem=problem), json.dumps([answer]), label, reason, t
        )

    a, b = rng.randint(40, 90), rng.randint(10, 30)
    items.append(
        math(
            f"A bakery in Tollerby sells {a} rye loaves on Monday and {b} more rye loaves on Tuesday than on "
            "Monday. How many rye loaves does it sell over the two days?",
            a + (a + b),
            "easy",
            "Two steps.",
            "math_two_step",
        )
    )
    base, per, free = rng.choice([60, 75]), rng.choice([2, 3]), rng.choice([10, 15])
    p1, p2, fuel = free + rng.randint(5, 20), free + rng.randint(5, 20), rng.choice([12, 18])
    total = 2 * base + per * ((p1 - free) + (p2 - free)) - 2 * fuel
    items.append(
        math(
            f"A courier is paid ${base} per day plus ${per} for each parcel after the first {free} parcels of "
            f"the day. On Monday she delivers {p1} parcels and on Tuesday {p2} parcels. Fuel costs her ${fuel} "
            "per day. What are her total earnings after fuel for the two days?",
            total,
            "hard",
            "Per-day threshold applied to each day separately, plus a per-day cost.",
            "math_courier",
        )
    )

    labels = "\n".join(HELPDESK_LABELS)

    def intent(query: str, gold: str, label: str, reason: str, t: str) -> Item:
        prompt = BANKING77.format(kind="staff helpdesk query", labels=labels, query=query)
        return Item("misc_classification", "exact_match", prompt, json.dumps([gold]), label, reason, t)

    items.append(
        intent(
            "My fuel card got rejected at the pump this morning, can someone check it?",
            "fuel_card_declined",
            "easy",
            "Direct match.",
            "intent_fuel_card",
        )
    )
    items.append(
        intent(
            "My manager already signed off the extra hours, but I typed 5 hours for Saturday instead of 8 and "
            "need that fixed before payroll runs.",
            "timesheet_correction",
            "hard",
            "Mentions approval and payroll, but the request is to correct entered hours.",
            "intent_timesheet",
        )
    )
    return items


# --- Build ------------------------------------------------------------------------------------------------

FAMILIES: list[tuple[str, Callable[[random.Random], list[Item]]]] = [
    ("qa", document_qa),
    ("cx", contracts),
    ("cd", coding),
    ("ms", misc),
]


def build(seed: int = SEED) -> list[dict[str, str]]:
    """All synthetic rows, in the dataset's CSV schema."""
    rng = random.Random(seed)
    rows = []
    for prefix, family in FAMILIES:
        for i, item in enumerate(family(rng), 1):
            rows.append(
                {
                    "id": f"syn-{prefix}-{i:02d}",
                    "use_case": item.use_case,
                    "source": "synthetic",
                    "source_id": f"{VERSION}:{item.template}",
                    "check_type": item.check_type,
                    "prompt": item.prompt,
                    "answer": item.answer,
                    "override_reason": "",
                    "label": item.label,
                    "label_reason": item.label_reason,
                }
            )
    return rows


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build()
    write(SYNTHETIC, rows)
    easy = sum(r["label"] == "easy" for r in rows)
    print(f"{len(rows)} synthetic rows ({easy} easy, {len(rows) - easy} hard) -> {SYNTHETIC.name}")


if __name__ == "__main__":
    main()
