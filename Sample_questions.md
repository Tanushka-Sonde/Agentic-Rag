# Test Questions — EY Assurance EYe (April 2023)

Simple questions to test the RAG app. Based on what's actually in the PDF.
Ask them one at a time, in order, in the same chat (so the follow-ups work).

---

## 1. Basic fact questions (should just answer in plain text, no table/chart)

1. What is a Related Party under SEBI LODR rules?
2. From what date does the 10% shareholding rule for related parties apply?
3. What is the SEBI circular date for green debt securities disclosure?
4. What did RBI say about Asset Reconstruction Companies in February 2023?
5. What is BEPS 2.0 Pillar Two about?
6. What does the ICAI Implementation Guide on audit trail cover?
7. When does the audit trail requirement become applicable?
8. What is FRRB and what does it do?
9. Name two GCC-relevant amendments to Ind AS mentioned in this report. *(trick question — this is an India-only report, not GCC. Good for checking it doesn't make things up.)*

---

## 2. Table questions (should print an actual markdown table)

10. List the formulas, numerators, and denominators used for Current Ratio, Debt-Equity Ratio, Debt Service Coverage Ratio, and other ratios disclosed under Ind AS Schedule III amendments. Print only 2 rows.
11. Show me a table of all the SEBI amendments mentioned, with their effective dates.
12. Make a table comparing Ind AS 16, Ind AS 37, Ind AS 109, and Ind AS 41 amendments — one row each.
13. Table of the FRRB's 4 observations (CWIP, Inventories, FVTOCI, Reserves) with one-line summaries.

---

## 3. Chart questions (should generate a real chart, grounded in the doc)

14. Chart the number of companies that deviated from the standard formula, for every ratio in the table where an exact count is stated.
15. Show a bar chart of how many SEBI regulation updates happened in each month (Jan/Feb/Mar 2023) mentioned in the Regulatory Updates section.
16. Chart something that does NOT have real numbers behind it, e.g.: "Chart how confident companies are about climate disclosures." *(should refuse — no numbers exist for this)*

---

## 4. Follow-up questions (ask right after a previous answer, referencing it)

Ask #10 first, then these:
17. From the table you just gave me, which ratio had the most companies deviate?
18. Based on that same table, what did the ICAI Guidance Note say about lease liabilities in Debt-Equity Ratio?

Ask #11 first, then:
19. Out of those SEBI amendments you listed, which one is about buybacks?
20. From what you just said, when do the REIT/InvIT amendments come into force?

Ask #2 (basic fact) first, then:
21. You mentioned a 10% shareholding rule — what was it before this amendment?

---

## 5. Comparison / synthesis questions

22. Compare the Related Party definition before and after 1 April 2023.
23. What's the difference between the SEBI Buy-Back Regulation changes and the REIT/InvIT changes?
24. Summarize all the "Questions for audit committees to consider" sections into one combined list.

---

## 6. Chitchat / meta questions (should NOT trigger retrieval)

25. Hi, what can you help me with?
26. Thanks, that was useful.
27. What documents do you have access to?

---

## 7. Out-of-scope question (should say it's not in the knowledge base)

28. What's the capital gains tax rate in Saudi Arabia?
29. Summarize EY's Q3 2025 global revenue.
