# Contributing

Bug reports, questions and suggestions are welcome as GitHub issues. Results on other
collections — where the rules work and where they fail — are especially useful; please include
the IIIF manifest or image URL and the command you ran.

For a pull request:

1. Install with `pip install -e ".[test]" pillow`
2. Run `pytest` and `python paper/reproduce.py`. Both must pass: a change to the rules must not
   change the paper's numbers silently. If a change is meant to change them, say so in the pull request.
3. Keep the rules in one place (`src/shohon_bbox/rules.py`); the command line, the notebook and the
   reproduction script all call them.
