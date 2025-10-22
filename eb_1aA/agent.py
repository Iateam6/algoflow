import logging
from datetime import datetime
from agents import Agent , Runner ,WebSearchTool ,FileSearchTool ,ModelSettings

async def initialize_agents(vector_store_id):
    """
    Initialize agents with the given vector store ID.
    """
    # Get today’s date in the desired format
    current_date = datetime.now().strftime("%B %#d, %Y")

    return {
        "I-140 Cover Letter": Agent(
            name="I-140 Cover Letter Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a I-140 Cover Letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ``` 
                Premium Processing  
                USCIS [Service Center Name]  
                [Street Address]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE:  Request for First Preference EB-1A I-140 with Sponsors 
                    Self‑Petitioner: [Beneficiary’s Full Name]  
                    Position/Title: [e.g., EVP of Technology]  

                Dear Immigration Officer:

                Please find enclosed the immigrant petition filed on behalf of [Beneficiary’s Full Name] as an EB-1A I-140 with Sponsors Alien of Extraordinary Ability.

                The following items are included in support of this petition:

                1.  A check for $[Amount] for the I‑907 premium processing fee  
                2.  A check for $[Amount] for the I‑140 filing fee  
                3.  Form I‑907, “Request for Premium Processing Service”  
                4.  Form G‑28, “Notice of Entry of Appearance as Attorney or Accredited Representative”  
                5.  Form I‑140, “Immigrant Petition for Alien Worker”  
                6.  [Beneficiary’s Last Name]’s biographical documents:  
                    a. Passport biographical page  
                    b. O‑1 approval notice  
                    c. Most recent I‑94  
                7.  Attorney’s letter of support  
                8.  Exhibit list  
                9.  Exhibits evidencing [Beneficiary’s Last Name]’s EB-1A I-140 with Sponsors credentials  

                We respectfully submit that the enclosed documentation establishes [Beneficiary’s Full Name]’s internationally recognized achievements and abilities as a leader in the [field] industry, and that he/she is among the small percentage of individuals who have risen to the top of his/her field.  [Beneficiary’s Last Name] therefore merits classification as an Alien of Extraordinary Ability.

                If you require any further information or documentation to support the attached petition, please do not hesitate to contact our office.

                Very truly yours,

                \_\_\_\_\_\_\_\_\_\_\_,  
                [Authorized Signatory’s Name]
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.

                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Support Letter": Agent(
            name="Support Letter Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a support letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                [Letterhead or Law Firm Name]  
                [Address Line 1]  
                [Address Line 2]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE: EB-1A I-140 with Sponsors Petition of [Beneficiary’s Full Name]  

                Dear Immigration Officer:

                Below please find our organized presentation of evidence in support of [Beneficiary’s Full Name]’s classification as an EB-1A I-140 with Sponsors nonimmigrant worker.  Each section corresponds to one of the key requirements for EB-1A I-140 with Sponsors classification:

                1. **Documentation of Receipt of Lesser Nationally or Internationally Recognized Prizes or Awards for Excellence**  
                – [Describe awards, dates, issuing organizations, and why they qualify.]

                2. **Documentation of Membership in Associations in the Field Which Require Outstanding Achievements**  
                – [List associations, membership criteria, and evidence of selection.]

                3. **Published Material About the Beneficiary in Professional or Major Trade Publications or Media**  
                – [Cite articles, dates, outlets, and excerpts relevant to the field.]

                4. **Evidence of Participation, Either Individually or on a Panel, as a Judge of the Work of Others**  
                – [Detail panels, dates, selection process, and scope of judging.]

                5. **Evidence of Original Contributions of Major Significance to the Field**  
                – [Summarize innovations, adoption by peers, citation metrics, and impact.]

                6. **Authorship of Scholarly Articles in Professional Journals or Other Major Media**  
                – [List publications, co‑authors, journal impact factors, and download/citation counts.]

                7. **Display of the Beneficiary’s Work at Artistic Exhibitions or Showcases**  
                – [Identify exhibitions or screenings, dates, venues, and audience reach.]

                8. **Evidence That the Beneficiary Has Performed in a Leading or Critical Role for Organizations or Establishments with a Distinguished Reputation**  
                – [Name companies or projects, describe role, and point to recognition.]

                9. **Evidence of Commercial Success in the Performing Arts, as Shown by Box‑Office Receipts or Record, Cassette, Compact Disc, or Video Sales**  
                – [Provide revenue figures, chart positions, and distributor confirmations.]

                **Conclusion & Prayer for Relief**  
                Based on the foregoing evidence (Sections 1–9), [Beneficiary’s Full Name] clearly meets the EB-1A I-140 with Sponsors criteria for specialty occupation.  We respectfully request that USCIS grant approval of the Form I‑140 petition.

                If you require further information or documentation, please contact our office.

                **very truly yours,**

                \_\_\_\_\_\_\_\_\_\_\_, 
                [Authorized Signatory’s Name]  
              
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Recommendation-Letter": Agent(
            name="Recommendation Letter Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Recommendation Letter for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                U.S. Citizenship and Immigration Services  
                U.S. Department of Homeland Security  

                Date: [YYYY‑MM‑DD]  

                RE:  [Recommender’s Name]’s Recommendation for EB-1A I-140 with Sponsors Petition of [Beneficiary’s Full Name]  

                Dear Sir or Madam:

                My name is [Recommender’s Name], [Title/Role] at [Organization(s)] and creator/executive producer of [List of Major Works].  I write in strong support of [Beneficiary’s Full Name]’s petition as an individual of extraordinary ability.

                Paragraph 1: Introduce your credentials and relationship to the Beneficiary.

                Paragraph 2: Summarize Beneficiary’s most significant U.S. achievements—lead roles, awards, box‑office metrics, publications, etc.

                Paragraph 3: Highlight Beneficiary’s industry impact (e.g., teaching, guest‑lecturing, innovation in distribution or production).

                Paragraph 4: Conclude that [Beneficiary’s Last Name] clearly qualifies for EB-1A I-140 with Sponsors classification and that U.S. interests will be served by granting the visa.  Offer to provide additional information if needed.

                **very truly yours,**

                \_\_\_\_\_\_\_\_\_\_\_,  
                [Recommender’s Printed Name]  

                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Exhibit List": Agent(
            name="Exhibit List Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                                                Exhibit List  
                                    Self‑Petitioner: [Beneficiary’s Full Name]  
                                    Position: [Beneficiary’s Position/Title]  

                Exhibit 1:  [Description of Exhibit 1]  
                Exhibit 2:  [Description of Exhibit 2]  
                Exhibit 3:  [Description of Exhibit 3]  
                Exhibit 4:  [Description of Exhibit 4]  
                Exhibit 5:  [Description of Exhibit 5]  
                Exhibit 6:  [Description of Exhibit 6]  
                Exhibit 7:  [Description of Exhibit 7]  
                Exhibit 8:  [Description of Exhibit 8]  
                Exhibit 9:  [Description of Exhibit 9]  
                Exhibit 10: [Description of Exhibit 10]  
                Exhibit 11: [Description of Exhibit 11]  
                Exhibit 12: [Description of Exhibit 12]  
                Exhibit 13: [Description of Exhibit 13]  
                Exhibit 14: [Description of Exhibit 14]  
                Exhibit 15: [Description of Exhibit 15]  
                Exhibit 16: [Description of Exhibit 16]  
                Exhibit 17: [Description of Exhibit 17]  
                Exhibit 18: [Description of Exhibit 18]  
                Exhibit 19: [Description of Exhibit 19]  
                Exhibit 20: [Description of Exhibit 20]  
                Exhibit 21: [Description of Exhibit 21]  
                Exhibit 22: [Description of Exhibit 22]  
                Exhibit 23: [Description of Exhibit 23]  
                Exhibit 24: [Description of Exhibit 24]  
                Exhibit 25: [Description of Exhibit 25]  
                Exhibit 26: [Description of Exhibit 26]  
                Exhibit 27: [Description of Exhibit 27]  
                Exhibit 28: [Description of Exhibit 28]  
                Exhibit 29: [Description of Exhibit 29]  
                Exhibit 30: [Description of Exhibit 30]  
                Exhibit 31: [Description of Exhibit 31]  
                Exhibit 32: [Description of Exhibit 32]  
                Exhibit 33: [Description of Exhibit 33]  
                Exhibit 34: [Description of Exhibit 34]

                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Evidence-Summary Chart": Agent(
            name="Evidence-Summary Chart Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Evidence Summary Chart for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                ## Evidence‑Summary Chart

                **1. Awards and Recognitions**  
                Document any nationally or internationally recognized awards that the beneficiary has received. Include details such as the awarding body, date, and context. Evidence may also include team awards, provided the beneficiary is a named recipient. Such awards demonstrate peer or institutional validation of excellence. If the award is part of a competition or grant selection process, explain its selectivity and prestige. This evidence helps satisfy the criterion under 8 CFR § 204.5(h)(3)(i) by validating recognized achievement in the field. Ensure documentation shows the beneficiary’s direct award receipt.

                **2. Membership in Associations**  
                Provide proof of membership in professional associations that require outstanding achievement for admission. Include documentation of the association’s selective criteria, the beneficiary’s membership status, and any leadership or committee roles held. Membership alone is assessable but becomes stronger when supplemented by active engagement such as organizing events or chapter leadership. This qualifies under the membership criterion and may bolster other criteria like critical roles or judging.

                **3. Published Material About the Beneficiary**  
                Include copies or clippings of media articles—preferably in major media outlets or professional trade journals—that discuss the beneficiary’s work in depth. The material must focus on the beneficiary and their contributions, not simply a passing mention. Provide full citations including title, author, date, publication, and translations if necessary. This aligns with the published material criterion and demonstrates external validation of the beneficiary’s influence.

                **4. Judging or Reviewing of Others’ Work**  
                Demonstrate instances where the beneficiary served as a peer reviewer, judging panel member, grant evaluator, or academic organizer. Include invitations, confirmation letters, or membership rosters listing the beneficiary’s participation. Even informal or one-time invites can qualify, provided they show recognition of expertise. This evidence supports the criterion under judging of others’ work. It further signals the beneficiary’s stature in their field.

                **5. Original Contributions of Major Significance**  
                Document contributions that have measurably impacted the field—such as new methodologies, patents, breakthrough research, or technological innovations. Provide supporting letters from experts describing the significance, peer citations, adoption rate, or real‑world impact. This category reflects the beneficiary’s role in advancing the discipline and satisfies the contributions criterion under the regulatory standard.

                **6. Authorship of Scholarly Articles**  
                List authored scholarly articles in reputable, peer‑reviewed journals or widely recognized professional publications. Include author order, impact factor of the journal, citation count, and audience reach. If the beneficiary was lead or sole author on high‑impact work, emphasize that. Even a single well‑cited article may qualify; but context on its prestige strengthens the case.

                **7. Leading or Critical Role in Distinguished Organizations**  
                Show evidence that the beneficiary held a leading or essential role in institutions or projects with reputational prominence. Documents may include organizational charts, reference letters, performance evaluations, or press coverage highlighting the beneficiary’s impact. Include details on how their contributions were central to organizational success. This supports the critical role criterion, signaling professional leadership.

                **8. High Salary or Commercial Success** *(if applicable)*  
                If relevant, provide documentation demonstrating that the beneficiary commands a high salary or achieved notable commercial success compared to peers. This may include salary records, contracts, revenue figures, or royalty statements in performing arts or innovations. Demonstrating remuneration above field norms supports eligibility under high‑salary or commercial success criteria.

                **9. Comparable Evidence** *(if standard criteria don’t directly fit)*  
                Where conventional categories may not apply—such as entrepreneurial achievements or field‑specific milestones—provide alternative evidence of comparable significance. Examples include unique funding awards, high‑profile conference presentations, invited keynote speeches, or patents obtained by startups. The updated Policy Manual explicitly allows such flexibility, particularly in STEM or innovative fields. Explain the relevance clearly.

                **Final Merits Commentary**  
                Summarize how the totality of submitted evidence demonstrates that the beneficiary is among the small percentage at the very top of their field with sustained national or international acclaim. Explain how meeting three or more regulatory criteria, along with the overall depth and quality of documentation, satisfies the two‑step adjudicative test under USCIS standards.
                

                ```

                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.

                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "RFE Response Brief": Agent(
            name="RFE Response Brief Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a RFE Response Brief for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                # Response to USCIS RFE – EB-1A I‑140 Petition for [Beneficiary’s Full Name]  
                **[Self‑Petitioner Name or Representative]**  
                **[Address]**  
                **[City, State, ZIP Code]**

                **Date:** _YYYY‑MM‑DD_

                **Subject:** Response to Request for Evidence (RFE) – EB‑1A Petition for [Beneficiary’s Full Name]

                **Dear USCIS Officer,**

                **Introduction & RFE Reference**  
                **Self‑Petitioner:** “[Beneficiary’s Full Name], as beneficiary and self‑petitioner, submits this response to the RFE issued regarding the Form I‑140 petition under the EB‑1A category.”  
                **RFE Details:**  
                   Receipt Number: _[Number]_  
                   RFE Issued: _[Date]_  
                   Response Deadline: _[Date]_  
                **Summary of USCIS Concerns:**  
                   1. Meeting at least three criteria under 8 C.F.R. § 204.5(h).  
                   2. Evidence of sustained national or international acclaim.  
                   3. Proof of intent to continue work in the field of extraordinary ability.

                **Response to USCIS Concerns**  
                **Concern 1: Satisfaction of Regulatory Criteria**  
                **USCIS Position:** “Insufficient documentation demonstrating satisfaction of a minimum of three alternative criteria for EB‑1A eligibility.”  
                **Rebuttal:**  
                We submit enhanced evidence addressing each criterion claimed: notably  
                • Receipt of lesser nationally‑recognized prizes (Ex. A);  
                • Authorship of scholarly articles in reputable journals (Ex. B);  
                • Original contributions of major significance supported by expert letters (Ex. C).  
                These documents are mapped precisely to each regulatory criterion under 8 C.F.R. § 204.5(h) and include explanatory summaries.

                **Concern 2: Sustained National or International Acclaim**  
                **USCIS Position:** “Documents do not clearly establish sustained acclaim or recognition at the top of the field.”  
                **Rebuttal:**  
                We include independent expert letters from recognized authorities in the field, highlighting the beneficiary’s international prominence, citation record, presentation invitations, and media coverage (Ex. D). Publications and citation metrics are arrayed chronologically to show consistent recognition over time in line with AAO precedent.  

                **Concern 3: Intent to Continue Work in Expertise**  
                **USCIS Position:** “Lack of evidence confirming that the beneficiary intends to continue working in the field of extraordinary ability in the United States.”  
                **Rebuttal:**  
                We have enclosed a detailed statement by the beneficiary outlining future research or professional plans, supported by letters from collaborators or institutions (Ex. E), documenting concrete plans to continue impactful work in the same field in the U.S. This satisfies the continuing‐work requirement and aligns with USCIS RFE guidance.

                **Additional Legal References**  
                We cite INA § 203(b)(1)(A) and 8 C.F.R. § 204.5(h) as governing authority for extraordinary ability classification.

                **Conclusion & Request**  
                **Factual and Regulatory Basis:** “Based on the robust documentary evidence and expert declarations submitted, the beneficiary satisfies at least three regulatory criteria under 8 C.F.R. § 204.5(h), demonstrates sustained acclaim, and intends to continue in the field.”  
                **Request for Approval:** “The petitioner respectfully requests prompt approval of the EB‑1A I‑140 petition for [Beneficiary’s Full Name].”  
                **Point of Contact:** “Please contact **[Representative Name]**, **[Title/Role]**, at **[Phone Number]** or **[Email Address]** for any further information or documentation.

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_\_\_,  
                **[Representative Name], [Title]**  
                **[If self‑petition: Beneficiary Name]**

  

                ```

                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.

                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Demand Letter": Agent(
            name="Demand Letter Agent",
            instructions=(
                        f"""
                        Today’s date is {current_date}.  
                        You are tasked with generating a Demand Letter for an EB-1A I-140 with Sponsors visa application.

                        **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                        - Personal details of the beneficiary or the client.
                        - Employer details.
                        - Job description and duties.
                        - Required forms:
                            - Form G-1145, E-Notification of Application/Petition Acceptance
                            - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                            - Form I-907, Request for Premium Processing Service
                            - Form I-140, Immigrant Petition for Alien Worker
                        - Supporting documents:
                            - All degree certificates
                            - Awards and recognitions
                            - Degree evidence
                            - Birth certificate
                            - Form I-94 (Arrival/Departure Record)
                            - Form W-2/1099 (Wage and Tax Statements)
                            - Publications
                            - Membership in organizations
                            - Passport
                            - Visa pages
                            - Letters of recommendation
                            - Formation documents (e.g., Articles of Incorporation)
                            - Federal tax returns
                            - Media reports

                        **Step 2**: Use the following structure for the letter:
    
                        ```
                        # Demand for Adjudication Under the Mandamus Act and Administrative Procedure Act – [Beneficiary’s Full Name]  
                        **[Attorney’s Name]**  
                        **[Law Firm Name]**  
                        **[Street Address]**  
                        **[City, State, ZIP Code]**  
                        **[Phone Number]**  
                        **[Email Address]**  

                        **Date:** _YYYY‑MM‑DD_  

                        **Employer / Sponsor Contact:**  
                        - **Name:** _[Sponsor or Employer’s Name]_  
                        - **Title:** _[Title]_  
                        - **Company:** _[Sponsor/Company Name]_  
                        - **Address:** _[Street Address], City, State, ZIP Code_  

                        **RE:** _Demand for Adjudication under the Mandamus Act – [Beneficiary’s Full Name] (EB‑1A I‑140 with Sponsor)_  

                        ### Dear [Sponsor or Employer Name],  

                        **Introduction & Jurisdiction**  
                        - **Parties:** “This letter is submitted by **[Law Firm Name]** on behalf of **[Sponsor/Employer Name]** (the “Petitioner”) in support of its EB‑1A I‑140 petition for **[Beneficiary Name]** (the “Beneficiary”), including sponsor support documentation.”  
                        - **Procedural History:**  
                        - I‑140 Filed: _[Date]_; Receipt No.: _[Number]_  
                        - RFE Issued (if any): _[Date]_ → Response Filed: _[Date]_ (if applicable)  
                        - Current Delay: _[Number]_ days beyond USCIS’s published processing benchmarks or internal guidance  
                        - **Jurisdiction:** Demand is made under *28 U.S.C. § 1361* (mandamus) and *5 U.S.C. § 555(b)* (unreasonable delay) to compel adjudication of the EB‑1A I‑140 petition.  

                        **Factual Background**  
                        - **Sponsor Profile:** Description of the sponsor/employer’s industry, size, and role in supporting the beneficiary’s extraordinary achievements and ability.  
                        - **Beneficiary Credentials:** Highest degree, field, exceptional achievements, international recognition, published works, citations, awards, former visa/status history, and evidence of sustained acclaim.  
                        - **Position and Sponsorship Details:** Offered position title, SOC code (if applicable), wage level, job duties aligned with extraordinary ability, sponsor’s role in supporting continued acclaim or national/international impact.  
                        - **Key Dates (Timeline):**  
                        - • _[Date]_ – I‑140 Filed  
                        - • _[Date]_ – RFE Issued (Ex. B), if applicable  
                        - • _[Date]_ – RFE Response Filed (Ex. C), if applicable  
                        - • _Today’s Date_ – Over _[X]_ days past reasonable adjudication period  

                        **Legal Standard for Mandamus and APA**  
                        - **Clear Right:** Petitioner and beneficiary have an indisputable right to timely adjudication of the I‑140 under the INA and USCIS regulations.  
                        - **Non‑Discretionary Duty:** USCIS has a ministerial duty to adjudicate filed I‑140 petitions within a reasonable timeframe.  
                        - **No Adequate Alternative Remedy:** Administrative status inquiries and service requests are inadequate under existing case law.  
                        - **Governing Authority:** Clear statutory basis under *28 U.S.C. § 1361* and *5 U.S.C. § 555(b)*; courts regularly permit mandamus or APA relief for delayed petition adjudications.

                        **Demand for Relief**  
                        - **Relief Sought:** We hereby demand that USCIS render a final decision on the EB‑1A I‑140 petition for **[Beneficiary’s Full Name]**, sponsored by **[Sponsor/Employer]**, no later than **14 calendar days** from receipt of this letter.  
                        - **Consequences if Unresolved:**  
                        - **Sponsor Hardship:** Delay imperils the sponsor’s planned engagement with the beneficiary, interrupts project timelines, and risks reputational harm or contractual penalties _(~ $X/week of delay)_.  
                        - **Beneficiary Hardship:** Continued delay causes expiration of existing status, risk of unlawful presence, disruption to international recognition trajectory, and harm to dependents.  

                        **Prejudice & Hardship**  
                        - **Sponsor Impact:**  
                        - Financial losses: ~$[Amount] per week due to postponed contributions or engagements.  
                        - Operational setbacks: inability to leverage beneficiary’s expertise for ongoing or upcoming projects.  
                        - **Beneficiary Impact:**  
                        - Loss of ability to maintain work authorization; risk of falling out of status on _[Date]_.  
                        - Negative consequences to international collaborative work and family stability.  
                        - **Irreparable Injury:** Monetary damages cannot adequately compensate for loss of status or professional momentum; only judicial relief can ensure adjudication.  

                        **Conclusion & Next Steps**  
                        - **Final Demand:** “We request that USCIS issue a final adjudication decision no later than 14 days from service of this demand.”  
                        - **Service Confirmation:** “Please confirm receipt via email to **[Attorney’s Email]** or fax to **[Fax Number]**.”  
                        - **Litigation Warning:** “Absent timely action, we are prepared to file a Writ of Mandamus in the U.S. District Court for the District of **[District]**, seeking to compel adjudication and recover EAJA fees and costs.”  
                        - **Attorney Availability:** “**[Attorney Name]** is available to supply any additional information or documentation USCIS may require.”  

                        **Very truly yours,**  
                        \_\_\_\_\_\_\_\_\_\_\_,  
                        **[Attorney’s Full Name], Esq.**  
                        **[Law Firm Name]**
  

                        ```
                       step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                       Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                       Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                       Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                       Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                       Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                       Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.

                        """

            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Assessment Report": Agent(
            name="Assessment Report Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Assessment Report for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                # EB‑1A I‑140 Support Letter – Extraordinary Ability Petition for [Beneficiary’s Full Name]  
                **Self‑Petitioner / Sponsor: [Beneficiary’s Full Name]**  
                **[Petitioner’s Name if Sponsor Organization]**  
                **[Petitioner’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** Assessment Report for EB‑1A I‑140 Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Petitioner:** “[Beneficiary’s Full Name], acting as self‑petitioner (the ‘Petitioner’), hereby submits this Assessment Report in support of Form I‑140 under EB‑1A classification for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                **Purpose:** To present a detailed evaluation of extraordinary ability in the sciences/business/arts and demonstrate eligibility under INA § 203(b)(1)(A) via evidence of sustained national or international acclaim.  
                **Summary:** This dossier outlines the Beneficiary’s achievements aligned to at least three of the ten regulatory EB‑1A criteria, followed by a totality‑of‑evidence merits discussion.

                **Overview of Extraordinary Ability Profile**  
                **Professional Summary:**  
                Education, awards, high‑impact contributions, media recognition, expert recommendation letters—framed to satisfy criteria under 8 CFR 204.5(h).  
                Key professional milestones, original contributions of major significance, roles as judge or reviewer, evidence of membership in elite associations, leading roles in distinguished organizations, and enhanced remuneration (if applicable).

                **Self‑Petition & Sponsorship**  
                As permitted under EB‑1A, no employer job offer or Labor Certification is required. The Petitioner sponsors themselves, demonstrating continuing intent to work in their field of expertise in the U.S., supported by contracts, client engagements, or project plans.

                **Summary of Qualifications**  
                **Educational Background:**  
                Degree(s) (e.g. Ph.D., M.S., B.S.) in **[Field]** from **[Institution]** (Date), including honors and recognition.  
                **Professional Achievements & Honors:**  
                Major awards, recognition by professional bodies, invited presentations, elected positions, or patents/publications establishing industry leadership.  
                **Criteria‑Specific Evidence:**  
                Each qualifying criterion is substantiated with labeled exhibits: publications about the individual, judging panels, membership proofs, expert support letters, citation indices, and project leadership documentation (see Exhibit List).

                **Regulatory Criteria Analysis**  
                — **Part One: Initial Criteria (8 CFR 204.5(h))**:  
                At least three criteria are met, including but not limited to:  
                • Evidence of original contributions of major significance  
                • Authorship of scholarly articles in major media  
                • Performance in a leading or critical role in distinguished organizations  
                • Judging the work of others  
                • Membership in associations requiring outstanding achievements  
                — **Part Two: Final Merits Determination (Kazarian standard)**:  
                Review of the totality of evidence supports that the Beneficiary is among the very small percentage at the top of the field and has sustained acclaim.

                **Supporting Documentation & Organizational Compliance**  
                — **Form I‑140:** Completed edition (mm/dd/yy) submitted with all required sections.  
                — **Form I‑907 (if premium processing requested):** Submitted concurrently.  
                — **Recommendation Letters:** Independent, expert support letters detailing the Beneficiary’s exceptional achievements and field contributions .  
                — **Publications, Media, Awards:** Packaged with translated materials, highlighted citations, and tabs corresponding to relevant criteria .  
                — **Exhibit Index:** Tabbed and cross‑referenced to eligibility criteria and items in Exhibit List.

                **Conclusion & Request**  
                **Eligibility Confirmed:** “Based on the foregoing, the Beneficiary clearly satisfies the extraordinary ability standards under INA § 203(b)(1)(A), meeting multiple regulatory criteria with evidence of sustained national/international acclaim.”  
                **Approval Requested:** “The Petitioner respectfully requests prompt approval of the EB‑1A I‑140 petition for **[Beneficiary’s Name]**, and stands ready to respond to any further inquiries.”  
                **Point of Contact:** “Please direct any questions to **[Petitioner’s Representative Name]**, **[Title]**, at **[Phone]** or **[Email]**.”  

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Authorized Signatory’s Name], [Title]**  
                **[Company or Self‑Petitioner Name]**
                

                ```

                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.

                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Visa Application Summary Report": Agent(
            name="Visa Application Summary Report Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Assessment Report for an EB-1A I-140 with Sponsors visa application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - Form G-1145, E-Notification of Application/Petition Acceptance
                    - Form G-28 (Company), Notice of Entry of Appearance as Attorney or Accredited Representative
                    - Form I-907, Request for Premium Processing Service
                    - Form I-140, Immigrant Petition for Alien Worker
                - Supporting documents:
                    - All degree certificates
                    - Awards and recognitions
                    - Degree evidence
                    - Birth certificate
                    - Form I-94 (Arrival/Departure Record)
                    - Form W-2/1099 (Wage and Tax Statements)
                    - Publications
                    - Membership in organizations
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Formation documents (e.g., Articles of Incorporation)
                    - Federal tax returns
                    - Media reports

                **Step 2**: Use the following structure for the letter:
                ```
                # EB‑1A I‑140 Application Summary Report – [Beneficiary’s Full Name]  
                **[Sponsor or Petitioner Name]**  
                **[Sponsor Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** EB‑1A I‑140 Petition Summary Report – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Sponsor’s Name] (the “Petitioner”) submits this I‑140 petition under the EB‑1A (Alien of Extraordinary Ability) category on behalf of **[Beneficiary’s Full Name]** (the “Beneficiary”).”  
                **Category & Intent:** EB‑1A classification as an individual of extraordinary ability under INA § 203(b)(1)(A). The Beneficiary self‑petitioned or is sponsored to demonstrate sustained national or international acclaim and intent to continue work in the field of extraordinary ability.  

                **Summary of Evidence**  
                The evidence clearly establishes that the Beneficiary meets the EB‑1A regulatory standard. Academic records and credential evaluations confirm advanced degree(s) in _[Field]_, aligning directly with the field of expertise. Documentation satisfies at least three of the ten regulatory criteria under 8 CFR 204.5(h), such as awards of national/international significance, original contributions to the field, authorship in high‑impact publications, judging peer work, and leadership roles in distinguished organizations. Extensive recommendation letters, citation metrics, publication indexes, and media references corroborate acclaim. Expert testimonials highlight influence and impact. No labor certification is required or submitted in this category.

                **Supporting Documents & Forms Enclosed**  
                - Form I‑140, properly completed and signed by Petitioner  
                - Form G‑28, Notice of Entry of Appearance as Attorney (if applicable)  
                - I‑797 receipt notice from any prior I‑140 approval (if beneficiary benefiting from earlier priority date)  
                - Detailed exhibit list mapping evidence to each regulatory criterion satisfied  
                - Academic diplomas, transcripts, and credential evaluation reports  
                - Expert recommendation letters, publication lists, citation reports  
                - Evidence of awards, membership in associations requiring excellence, media coverage, judging records, salary or remuneration comparisons  
                - Supplemental original contributions and organizational leadership documentation  

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the totality of evidence, **[Beneficiary’s Full Name]** unquestionably meets the statutory and regulatory standard for classification as an Alien of Extraordinary Ability under EB‑1A.”  
                **Request for Approval:** “Petitioner respectfully requests that USCIS adjudicate Form I‑140 under the EB‑1A category promptly and grant the Immigrant Petition without delay.”  
                **Point of Contact:** “For any questions or requests for additional documentation, please contact **[Petitioner’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Authorized Signatory Name], [Title]**  
                **[Sponsor or Petitioner Company or Firm]**  

                ```

                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Supporting Evidence & Exhibits" section, list only the exhibits for which supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.

                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
    }

#Initialize agents with the given vector store ID function
async def generate_document(file_type, agents):
    """
    Generate a single document using the corresponding agent.

    """
    agent = agents.get(file_type)
    if agent:
        logging.info(f"Generating document: {file_type} using {agent.name}")
        # Simulate the agent's task (replace this with actual agent execution logic)
        result = await Runner.run(agent, file_type)  # Assuming `run()` is a synchronous method
        print(f"Generated {file_type}: {result}")
        return result
    else:
        logging.warning(f"No agent found for document type: {file_type}")
        return None
