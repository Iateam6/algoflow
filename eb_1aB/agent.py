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
                You are tasked with generating a petition cover letter for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

                **Step 2**: Use the following structure for the letter:
                ``` 
                Premium Processing  
                USCIS [Service Center Name]  
                [Street Address]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE:  Request for First Preference EB-1A I-140 & I-485 with Sponsors 
                    Self‑Petitioner: [Beneficiary’s Full Name]  
                    Position/Title: [e.g., EVP of Technology]  

                Dear Immigration Officer:

                Please find enclosed the immigrant petition filed on behalf of [Beneficiary’s Full Name] as an EB-1A I-140 & I-485 with Sponsors Alien of Extraordinary Ability.

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
                9.  Exhibits evidencing [Beneficiary’s Last Name]’s EB-1A I-140 & I-485 with Sponsors credentials  

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
                You are tasked with generating an employer support letter for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

                **Step 2**: Use the following structure for the letter:
                ```
                [Letterhead or Law Firm Name]  
                [Address Line 1]  
                [Address Line 2]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE: EB-1A I-140 & I-485 with Sponsors Petition of [Beneficiary’s Full Name]  

                Dear Immigration Officer:

                Below please find our organized presentation of evidence in support of [Beneficiary’s Full Name]’s classification as an EB-1A I-140 & I-485 with Sponsors nonimmigrant worker.  Each section corresponds to one of the key requirements for EB-1A I-140 & I-485 with Sponsors classification:

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
                Based on the foregoing evidence (Sections 1–9), [Beneficiary’s Full Name] clearly meets the EB-1A I-140 & I-485 with Sponsors criteria for specialty occupation.  We respectfully request that USCIS grant approval of the Form I‑140 petition.

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
                You are tasked with generating an employer support letter for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

                **Step 2**: Use the following structure for the letter:
                ```
                U.S. Citizenship and Immigration Services  
                U.S. Department of Homeland Security  

                Date: [YYYY‑MM‑DD]  

                RE:  [Recommender’s Name]’s Recommendation for EB-1A I-140 & I-485 with Sponsors Petition of [Beneficiary’s Full Name]  

                Dear Sir or Madam:

                My name is [Recommender’s Name], [Title/Role] at [Organization(s)] and creator/executive producer of [List of Major Works].  I write in strong support of [Beneficiary’s Full Name]’s petition as an individual of extraordinary ability.

                Paragraph 1: Introduce your credentials and relationship to the Beneficiary.

                Paragraph 2: Summarize Beneficiary’s most significant U.S. achievements—lead roles, awards, box‑office metrics, publications, etc.

                Paragraph 3: Highlight Beneficiary’s industry impact (e.g., teaching, guest‑lecturing, innovation in distribution or production).

                Paragraph 4: Conclude that [Beneficiary’s Last Name] clearly qualifies for EB-1A I-140 & I-485 with Sponsors classification and that U.S. interests will be served by granting the visa.  Offer to provide additional information if needed.

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
                You are tasked with generating an employer support letter for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

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
                You are tasked with generating an employer support letter for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

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
                You are tasked with generating a response brief to rebut USCIS concerns regarding EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

                **Step 2**: Use the following structure for the letter:
                ```
                # Response to USCIS RFE – EB-1A I-140 & I-485 Petition for [Beneficiary’s Full Name]
                **[Sponsor’s Full Name]**  
                **[Sponsor’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_

                **Subject:** Response to Request for Evidence (RFE) – [Beneficiary’s Full Name]

                Dear USCIS Officer,

                ## Introduction & RFE Reference

                **Parties & Purpose:**  
                “**[Sponsor’s Full Name]** (the “Petitioner”) submits this Response to the Request for Evidence (RFE) issued in connection with its EB-1A I-140 & I-485 petition for **[Beneficiary’s Full Name]** (the “Beneficiary”).”

                **RFE Details:**  
                Receipt Number: _[Number]_  
                RFE Issued: _[Date]_  
                Response Deadline: _[Date]_

                **Summary of USCIS Concerns:**  
                1. Question as to whether the Beneficiary meets at least three of the ten regulatory criteria for EB-1A classification.  
                2. Insufficient evidence of the Beneficiary’s sustained national or international acclaim.  
                3. Lack of documentation demonstrating that the Beneficiary will continue to work in their area of extraordinary ability in the United States.

                ## Rebuttal to USCIS Concerns

                ### Concern 1: EB-1A Regulatory Criteria

                **USCIS Position:** “The evidence provided does not establish that the Beneficiary meets at least three of the ten regulatory criteria for EB-1A classification.”

                **Rebuttal:**  
                - **Criterion 1: Receipt of Lesser Nationally or Internationally Recognized Prizes or Awards for Excellence**  
                - Submitted certified copies of awards and recognitions received by the Beneficiary, including [List of Awards] (Ex. A).  
                - Provided letters from recognized experts in the field attesting to the significance of these awards (Ex. B).  

                - **Criterion 2: Membership in Associations in the Field Which Demand Outstanding Achievement of Their Members**  
                - Included evidence of the Beneficiary’s membership in [List of Associations] (Ex. C).  
                - Provided documentation outlining the criteria for membership and how the Beneficiary meets these criteria (Ex. D).  

                - **Criterion 3: Published Material About the Beneficiary in Professional or Major Trade Publications or Other Major Media**  
                - Submitted copies of articles and publications featuring the Beneficiary, including [List of Publications] (Ex. E).  
                - Provided translations of non-English publications (Ex. F).  

                - **Criterion 4: Participation as a Judge of the Work of Others in the Same or Allied Field**  
                - Included evidence of the Beneficiary’s role as a judge or panelist at [List of Events] (Ex. G).  
                - Provided letters from event organizers confirming the Beneficiary’s participation (Ex. H).  

                - **Criterion 5: Original Contributions of Major Significance to the Field**  
                - Detailed the Beneficiary’s contributions to [Specific Field], including [Description of Contributions] (Ex. I).  
                - Provided letters from experts attesting to the significance of these contributions (Ex. J).  

                ### Concern 2: Sustained National or International Acclaim

                **USCIS Position:** “The evidence does not demonstrate that the Beneficiary has sustained national or international acclaim in their field.”

                **Rebuttal:**  
                - Provided a comprehensive timeline of the Beneficiary’s career, highlighting key achievements and recognitions (Ex. K).  
                - Included letters from industry leaders and experts attesting to the Beneficiary’s sustained acclaim (Ex. L).  

                ### Concern 3: Intent to Continue Work in the Area of Extraordinary Ability

                **USCIS Position:** “The petition does not indicate that the Beneficiary has prearranged commitments for working in this field.”

                **Rebuttal:**  
                - Submitted a detailed employment offer letter from [Employer’s Name], outlining the terms of employment and the nature of the work to be performed (Ex. M).  
                - Provided a copy of the signed contract between the Beneficiary and [Employer’s Name] (Ex. N).  

                ## Additional Legal Authority

                - Referenced INA § 203(b)(1)(A) and 8 C.F.R. § 204.5(h) regarding the classification of aliens of extraordinary ability.

                ## Conclusion & Request

                **Eligibility Reaffirmed:**  
                “Based on the expanded evidence and legal authorities cited, the Beneficiary clearly meets the requirements for classification as an alien of extraordinary ability under INA § 203(b)(1)(A).”

                **Request for Adjudication:**  
                “Petitioner respectfully requests that USCIS approve the EB-1A I-140 & I-485 petition for **[Beneficiary’s Full Name]** promptly and notify the Petitioner by email at **[Email Address]**.”

                **Point of Contact:**  
                “For any further questions or documentation requests, please contact **[Sponsor’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”

                Very truly yours,  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Sponsor’s Representative Name], [Title]**  
                **[Sponsor’s Organization Name]**


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
                        You are tasked with generating a Demand Letter for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                            - Form I-94 (Arrival/Departure Record)
                            - Passport
                            - Visa pages
                            - Letters of recommendation
                            - Media reports
                            - Social Security card
                            - Permanent Resident Card (Green Card)
                            - Petitioner’s pay stubs

                        **Step 2**: Use the following structure for the letter:
    
                        ```
                        # Demand for Adjudication Under the Mandamus Act and Administrative Procedure Act – [Beneficiary’s Full Name]

                        **[Attorney’s Name]**  
                        **[Law Firm Name]**  
                        **[Street Address]**  
                        **[City, State, ZIP Code]**  
                        **[Phone Number]**  
                        **[Email Address]**  

                        **Date:** _YYYY-MM-DD_

                        **Employer Contact:**  
                        - **Name:** _[Employer’s Name]_  
                        - **Title:** _[Title]_  
                        - **Company:** _[Company Name]_  
                        - **Address:** _[Street Address], City, State, ZIP Code_

                        **RE:** _Demand for Adjudication under the Mandamus Act – [Beneficiary’s Full Name]_

                        ### Dear [Employer’s Name],

                        **Introduction & Jurisdiction**

                        - **Parties:** “This letter is submitted by **[Law Firm Name]** on behalf of **[Employer Name]** (the “Petitioner”) in support of its EB-1A I-140 and I-485 petition for **[Beneficiary Name]** (the “Beneficiary”).”
                        - **Procedural History:**
                        - I-140 Filed: _[Date]_; Receipt No.: _[Number]_
                        - I-485 Filed: _[Date]_; Receipt No.: _[Number]_
                        - RFE Issued: _[Date]_ → Response Filed: _[Date]_
                        - Current Delay: _[Number]_ days beyond USCIS’s 60-day guideline
                        - **Jurisdiction:** Demand is made under *28 U.S.C. § 1361* (mandamus) and *5 U.S.C. § 555(b)* (unreasonable delay).

                        **Factual Background**

                        - **Employer Profile:** Industry, size, nature of business, and critical need for Beneficiary’s skills.
                        - **Beneficiary Credentials:** Degree, field, years of experience, prior visa status.
                        - **Position Details:** Title, SOC code, wage level, project description, worksite location(s).
                        - **Key Dates (Timeline):**
                        - • _[Date]_ – I-140 Filed
                        - • _[Date]_ – I-485 Filed
                        - • _[Date]_ – RFE Issued (Ex. B)
                        - • _[Date]_ – RFE Response Filed (Ex. C)
                        - • _Today’s Date_ – Over _[X]_ days past target

                        **Legal Standard for Mandamus**

                        - **Clear Right:** Petitioner’s indisputable right to a decision.
                        - **Non-Discretionary Duty:** USCIS must adjudicate within reasonable time.
                        - **No Adequate Alternative:** Status inquiry or service request is insufficient.
                        - **Agency Guidelines:** USCIS aims to resolve RFEs within 60 days (see July 17, 2017 Policy Memo).

                        **Demand for Relief**

                        - **Relief Sought:** Adjudication of the EB-1A I-140 and I-485 petitions within **14 days** of receipt.
                        - **Statutory Authority:** *28 U.S.C. § 1361*; *5 U.S.C. § 555(b)*.
                        - **Consequences if Unresolved:**
                        - **Employer Hardship:** Project delays, breach of contract, revenue loss (≈ $X/week).
                        - **Beneficiary Hardship:** Loss of work authorization on _[Date]_, family disruption.

                        **Prejudice & Hardship**

                        - **Employer Impact:**
                        - Financial loss: ~$[Amount] per week of delay.
                        - Operational setbacks: missed deadlines, client penalties.
                        - **Beneficiary Impact:**
                        - Authorized stay expires on _[Date]_; risk of unlawful presence.
                        - Dependents’ schooling and stability jeopardized.
                        - **Irreparable Injury:** Monetary damages inadequate; only mandamus will remedy.

                        **Conclusion & Next Steps**

                        - **Final Demand:** “We request USCIS issue a final decision no later than 14 days from service.”
                        - **Service Confirmation:** “Please confirm receipt via email to **[Attorney’s Email]** or fax to **[Fax Number]**.”
                        - **Litigation Warning:** “Absent timely action, we will file a Writ of Mandamus in the U.S. District Court for the District of **[District]**, and seek EAJA fees and costs.”
                        - **Attorney Availability:** “[Attorney Name] is available to provide any further information USCIS may require.”

                        **very truly yours,**  
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
                You are tasked with generating a Assessment Report for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

                **Step 2**: Use the following structure for the letter:
                ```
                # EB-1A I-140 & I-485 Petition Support Letter – [Beneficiary’s Full Name]

                **[Sponsor’s Name]**  
                **[Sponsor’s Address]**  
                **[City, State, ZIP Code]**

                **Date:** _YYYY-MM-DD_

                **Subject:** Assessment Report for EB-1A I-140 & I-485 Petition – [Beneficiary’s Full Name]

                **Dear USCIS Officer,**

                **Introduction**  
                **Parties:** “[Sponsor’s Name] (the “Petitioner”) submits this Assessment Report in support of its EB-1A I-140 & I-485 petition for **[Beneficiary’s Full Name]** (the “Beneficiary”).”  
                **Purpose:** To provide a comprehensive evaluation of the Beneficiary’s extraordinary ability and demonstrate statutory eligibility under INA § 203(b)(1)(A).  
                **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Wage Level: _[Level]_; Location(s): _[City, State]_.

                **Sponsor & Position Description**  
                **Sponsor Profile:**  
                - Industry, size, years in operation.  
                - Core business activities and key clients.  
                - Why specialized expertise of the Beneficiary is essential to operations/projects.  
                **Job Duties & Requirements:**  
                - Detailed list of primary and ancillary duties.  
                - Minimum education and experience prerequisites.  
                - Specialized tools, methodologies, software, or processes required.

                **Summary of Qualifications**  
                **Educational Background:**  
                - Degree(s) earned (e.g., B.S., M.S., Ph.D.) in _[Field]_ from _[Institution]_ (Date).  
                - Honors, thesis title, accredited status of institution.  
                **Professional Experience:**  
                - _[Years]_ years at _[Company]_ as _[Role]_; key achievements and project summaries.  
                - Prior EB-1A or other visa status (if applicable) with USCIS receipt numbers and approval dates.  
                **Specialized Knowledge & Skills:**  
                - Technical proficiencies (software, programming languages, analytical techniques).  
                - Certifications, published papers, patents, or speaking engagements.  
                - Unique contributions to past or ongoing projects demonstrating non-routine expertise.

                **Alignment with Regulatory Criteria**  
                **“Extraordinary Ability” Analysis (INA § 203(b)(1)(A)):**  
                - Explain how the Beneficiary meets at least three of the ten regulatory criteria.  
                - Provide evidence supporting each criterion.  
                **“Beneficiary’s Qualifications” Analysis:**  
                - Connect each degree and experience bullet to a corresponding duty or requirement.  
                - Cite USCIS policy memoranda or AAO decisions where similar profiles were approved.

                **Legal & Procedural Compliance**  
                **Labor Condition Application (LCA):** LCA certified on _[Date]_; wage level and worksite locations match.  
                **Public Access File:** Confirm availability of required documentation at worksite.  
                **Dependents & Maintenance of Status:** Brief note on any accompanying H-4 or E-dependent filings.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing, the Beneficiary clearly meets the educational and experiential requirements for the specialty occupation.”  
                **Favorable Adjudication Sought:** “Petitioner respectfully requests that USCIS approve the EB-1A I-140 & I-485 petition for **[Beneficiary’s Full Name]** promptly, in accordance with INA § 203(b)(1)(A).”  
                **Point of Contact:** “Please direct any questions or requests for additional information to **[Sponsor’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Sponsor’s Representative Name], [Title]**  
                **[Company Name]**
 

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
                You are tasked with generating a Visa Application Summary Report for an EB-1A I-140 & I-485 with Sponsors visa application.

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
                    - Form I-94 (Arrival/Departure Record)
                    - Passport
                    - Visa pages
                    - Letters of recommendation
                    - Media reports
                    - Social Security card
                    - Permanent Resident Card (Green Card)
                    - Petitioner’s pay stubs

                **Step 2**: Use the following structure for the letter:
                ```
                # EB-1A I-140 & I-485 Visa Application Summary Report – [Beneficiary’s Full Name]
                **[Sponsor’s Name]**  
                **[Sponsor’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_

                **Subject:** Visa Application Summary Report – [Beneficiary’s Full Name]

                **Dear USCIS Officer,**

                **Introduction**  
                **Parties & Purpose:** “[Sponsor’s Name] (the “Petitioner”) submits this Visa Application Summary Report in support of its EB-1A I-140 petition and Form I-485 Adjustment of Status application for [Beneficiary’s Full Name] (the “Beneficiary”).”  
                **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_.  
                **Objective:** Provide a concise narrative of the key evidence establishing the Beneficiary’s eligibility under INA § 203(b)(1)(A) and § 245(a).

                **Summary of Evidence**  
                The record demonstrates beyond question that the Beneficiary possesses the required extraordinary ability in the field of [Field], as evidenced by the following:

                - **Academic Credentials:** Certified diploma and official transcripts from [Institution] establish the Bachelor’s/Master’s degree in [Field] directly related to the position’s theoretical and practical demands.

                - **Professional Experience:** Detailed employment verification letters and the resume illustrate [X] years of progressive responsibility in [Specialty Area], including leadership of complex projects and demonstrated proficiency with [Key Tools/Technologies].

                - **Awards and Recognitions:** Documentation of nationally or internationally recognized prizes or awards for excellence in [Field], confirming the Beneficiary’s exceptional achievements.

                - **Memberships:** Evidence of membership in associations in the field which demand outstanding achievement of their members, underscoring the Beneficiary’s standing in the field.

                - **Publications and Media Coverage:** Published material about the Beneficiary in professional or major trade publications or other major media, highlighting the Beneficiary’s contributions and recognition in the field.

                - **Judging Roles:** Documentation of participation as a judge of the work of others in the same or allied field, demonstrating the Beneficiary’s expertise and recognition by peers.

                - **Original Contributions:** Evidence of original scientific, scholarly, artistic, athletic, or business-related contributions of major significance to the field, illustrating the Beneficiary’s impact and leadership.

                - **Authorship:** Copies of scholarly articles authored by the Beneficiary in professional or major trade publications or other major media, showcasing the Beneficiary’s thought leadership and influence.

                - **Exhibitions and Showcases:** Evidence of display of work at artistic exhibitions or showcases, if applicable, highlighting the Beneficiary’s visibility and acclaim in the field.

                - **Leading Roles:** Letters or documents confirming the Beneficiary’s performance of a leading or critical role in distinguished organizations, underscoring the Beneficiary’s leadership and influence.

                - **Remuneration:** Salary records or compensation statements indicating high salary or other significantly high remuneration in relation to others in the field, reflecting the Beneficiary’s exceptional qualifications and demand.

                - **Commercial Success:** Sales records, box office receipts, or other evidence of commercial success in the performing arts, if applicable, demonstrating the Beneficiary’s impact and popularity.

                - **Labor Condition Application (LCA):** Certified LCA, confirming prevailing wage compliance and worksite details.

                - **Credential Evaluation Reports:** Independent credential evaluations affirming the equivalency of the Beneficiary’s foreign degrees to U.S. degrees.

                - **Organizational Chart:** Comprehensive organizational chart contextualizing the Beneficiary’s unique contributions to critical client deliverables, underscoring the absence of comparably qualified U.S. applicants.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing, the Beneficiary incontrovertibly satisfies all statutory and regulatory requirements for EB-1A classification.”  
                **Request for Adjudication:** “Petitioner respectfully requests that USCIS approve the EB-1A I-140 petition and Form I-485 Adjustment of Status application for [Beneficiary’s Full Name] without delay.”  
                **Point of Contact:** “For any questions or additional documentation, please contact [Sponsor’s Representative Name], [Title], at [Phone Number] or [Email Address].”

                **Very truly yours,**  
                _\_\_\_\_\_\_\_\_\_\_\_, 
                [Sponsor’s Representative Name], [Title]  
                [Company Name]

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
