"""Fixed questions used by the evaluation harness."""

CASES = [
    {
        "id": "expense_deadline",
        "question": "What is the expense report deadline?",
        "expected_source": "02_expense_reimbursement_policy.md",
        "expected_phrase": "30 calendar days",
        "accept": r"(?<!\d)30 calendar days",
        "reject": r"do not file within (?:\*\*)?30 calendar days",
    },
    {
        "id": "expense_receipt",
        "question": "When is an itemized expense receipt required?",
        "expected_source": "02_expense_reimbursement_policy.md",
        "expected_phrase": "$25 or more",
        "accept": r"(?<![\d$,])\$25 or more",
        "reject": r"not required for expenses of (?:\*\*)?\$25 or more",
    },
    {
        "id": "expense_meals",
        "question": "What is the daily expense meal cap while traveling?",
        "expected_source": "02_expense_reimbursement_policy.md",
        "expected_phrase": "$65 per person per day",
        "accept": r"(?<![\d$,])\$65 per person per day",
        "reject": r"not reimbursable up to (?:\*\*)?\$65 per person per day",
    },
    {
        "id": "expense_finance_approval",
        "question": "When does a purchase need Finance approval in the expense policy?",
        "expected_source": "02_expense_reimbursement_policy.md",
        "expected_phrase": "$2,000",
        "accept": r"(?<![\d$,])\$2,000",
        "reject": r"not required above (?:\*\*)?\$2,000",
    },
    {
        "id": "pto_allowance",
        "question": "How many PTO days do full-time employees receive each year?",
        "expected_source": "03_paid_time_off_policy.md",
        "expected_phrase": "20 days (160 hours)",
        "accept": r"20 days \(160 hours\)",
        "reject": r"(?:do not|don't|never|not) receive (?:\*\*)?20 days \(160 hours\)",
    },
    {
        "id": "pto_carryover",
        "question": "How many PTO hours can an employee carry over?",
        "expected_source": "03_paid_time_off_policy.md",
        "expected_phrase": "40 hours",
        "accept": r"(?<!\d)40 hours",
        "reject": (
            r"140 hours|(?:may not|must not|cannot|can not|do not|don't|never) "
            r"carry over(?: up to)? (?:\*\*)?(?<!\d)40 hours"
        ),
    },
    {
        "id": "pto_notice",
        "question": (
            "How far in advance should an employee request PTO "
            "of three or more consecutive business days?"
        ),
        "expected_source": "03_paid_time_off_policy.md",
        "expected_phrase": "two weeks in advance",
        "accept": r"two weeks in advance",
        "reject": (
            r"(?:do not|don't|never|not) (?:have to |need to )?request "
            r"(?:pto |planned leave )?(?:at least )?two weeks in advance"
        ),
    },
    {
        "id": "remote_wifi",
        "question": "How should a remote employee connect when using public Wi-Fi?",
        "expected_source": "01_remote_work_policy.md",
        "expected_phrase": "company-approved secure access method",
        "accept": r"company-approved secure access method",
        "reject": (
            r"(?:do not|don't|never) connect through "
            r"the company-approved secure access method"
        ),
    },
]
