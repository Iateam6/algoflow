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
        "Petition Cover Letter": Agent(
            name="Petition Cover Letter Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a cover letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):

                ```
                Premium Processing  
                USCIS [Service Center Name]  
                [Street Address]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE:  Request for First Preference Reentry Permit 
                    Self‑Petitioner: [Beneficiary’s Full Name]  
                    Position/Title: [e.g., EVP of Technology]  

                Dear Immigration Officer:

                Please find enclosed the immigrant petition filed on behalf of [Beneficiary’s Full Name] as an Reentry Permit Alien of Extraordinary Ability.

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
                9.  Exhibits evidencing [Beneficiary’s Last Name]’s Reentry Permit credentials  

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
            model_settings=ModelSettings(temperature=0.9),
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
                You are tasked with generating a support letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                [Letterhead or Law Firm Name]  
                [Address Line 1]  
                [Address Line 2]  
                [City, State ZIP]  

                Date: [YYYY‑MM‑DD]  

                RE:Reentry Permit Petition of [Beneficiary’s Full Name]  

                Dear Immigration Officer:

                Below please find our organized presentation of evidence in support of [Beneficiary’s Full Name]’s classification as an Alien of Extraordinary Ability under 8 C.F.R. § 204.5(h).  Each section corresponds to one of the nine regulatory criteria:

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
                Based on the foregoing evidence (Sections 1–9), [Beneficiary’s Full Name] clearly meets the Reentry Permit criteria for extraordinary ability.  We respectfully request that USCIS grant approval of the Form I‑140 petition.

                If you require further information or documentation, please contact our office.

                **Sincerely,**

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
        "Recommendation Letter": Agent(
            name="Recommendation Letter Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Recommendation Letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                U.S. Citizenship and Immigration Services  
                U.S. Department of Homeland Security  

                Date: [YYYY‑MM‑DD]  

                RE:  [Recommender’s Name]’s Recommendation for Reentry Permit Petition of [Beneficiary’s Full Name]  

                Dear Sir or Madam:

                My name is [Recommender’s Name], [Title/Role] at [Organization(s)] and creator/executive producer of [List of Major Works].  I write in strong support of [Beneficiary’s Full Name]’s petition as an individual of extraordinary ability.

                Paragraph 1: Introduce your credentials and relationship to the Beneficiary.

                Paragraph 2: Summarize Beneficiary’s most significant U.S. achievements—lead roles, awards, box‑office metrics, publications, etc.

                Paragraph 3: Highlight Beneficiary’s industry impact (e.g., teaching, guest‑lecturing, innovation in distribution or production).

                Paragraph 4: Conclude that [Beneficiary’s Last Name] clearly qualifies for EB‑1A and that U.S. interests will be served by granting the visa.  Offer to provide additional information if needed.

                Sincerely,

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
                You are tasked with generating a Exhibit List in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
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
        "RFE Response Brief": Agent(
            name="RFE Response Brief Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a RFE Response Brief in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Response to USCIS RFE – Reentry Permit for [Applicant’s Full Name]
                **Consulate/USCIS Lockbox Name**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Response to Request for Evidence (RFE) – Reentry Permit for [Applicant’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction & RFE Reference**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”) submits this Response to the Request for Evidence (RFE) issued in connection with the Reentry Permit application (Form I‑131).”  
                **RFE Details:**  
                - Receipt Number: _[Number]_  
                - RFE Issued: _[Date]_  
                - Response Deadline: _[Date]_  
                **Summary of USCIS Concerns:**  
                1. Eligibility under INA § 216(a) and statutory basis.  
                2. Evidence of continuous U.S. residence and non‑abandonment.  
                3. Adequacy of documentation supporting purpose and duration of travel.  

                **Rebuttal to USCIS Concerns**  
                **Concern 1: Statutory Eligibility**  
                - **USCIS Position:** “Applicant may not qualify under INA § 216(a) for a reentry permit.”  
                - **Rebuttal:**  
                The Applicant is a Lawful Permanent Resident (A‑Number [#]), holding a valid PR card through [date], and meets all criteria under INA § 216(a) for issuance of a reentry permit to preserve residence abroad.  

                **Concern 2: Continuous U.S. Residence**  
                - **USCIS Position:** “Insufficient evidence that Applicant maintains continuous U.S. residence.”  
                - **Rebuttal:**  
                Enclosed are: Form I‑94 copy (Ex. B), tax returns for the past [#] years (Ex. G), employment verification letters (Ex. H), and evidence of property and family ties in the U.S. demonstrating non‑abandonment.  

                **Concern 3: Purpose and Duration of Travel**  
                - **USCIS Position:** “Purpose of travel and intended length abroad not adequately supported.”  
                - **Rebuttal:**  
                Applicant’s detailed travel explanation letter (Ex. D) outlines an intended absence of approximately [# months/years] for [reason: e.g., family emergency, employment assignment].  Return‑ticket reservation or proof of ongoing ties (Ex. E) further demonstrates intent to return.  

                **Additional Legal Authority:**  
                Cite INA § 216(a) permitting reentry permits for LPRs; 8 C.F.R. § 223.2 outlining evidentiary requirements.  

                **Supporting Evidence & Exhibits**  
                - **Exhibit A:** Copy of Permanent Resident Card  
                - **Exhibit B:** Copy of Form I‑94  
                - **Exhibit C:** Two passport‑style photos (USCIS specifications)  
                - **Exhibit D:** Letter from Applicant explaining purpose and duration of travel  
                - **Exhibit E:** Return‑ticket reservation or proof of U.S. ties  
                - **Exhibit F:** Prior reentry permits or boarding foils (if any)  
                - **Exhibit G:** Federal tax returns for years [YYYY–YYYY]  
                - **Exhibit H:** Employment verification letter(s)  

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing evidence and statutory authority, the Applicant clearly qualifies for issuance of a Reentry Permit under INA § 216(a).”  
                **Request for Adjudication:** “Applicant respectfully requests prompt adjudication and approval of the Reentry Permit application.”  
                **Point of Contact:** “For any further questions or additional documentation, please contact [Attorney/Representative Name], [Title], at [Phone Number] or [Email Address].”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_, 
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  
 

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
                    You are tasked with generating a Demand Letter in support of a Reentry Permit (Form I-131) application under Consular Processing.

                    **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                    - Applicant personal details (name, A‑number, date of birth, country of nationality)
                    - Consulate/USCIS lockbox address and beneficiary’s mailing address
                    - Travel dates, intended length of absence, and reason for travel
                    - Attorney/representative details (if any)

                    **Required Forms**  
                    • Form I‑131, Application for Travel Document (USCIS)  
                    • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                    • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                    • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                    **Supporting Documents**  
                    **Must Have**  
                    • Copy of Permanent Resident Card (USCIS)  
                    • Two passport‑style photos (USCIS photo specs)  
                    • Form I‑94 copy (CBP)  
                    • Passport biographic and visa pages (State/CBP)  
                    • Explanation letter from applicant detailing purpose and length of intended absence  
                    **As Available**  
                    • Return‐ticket reservation or proof of strong ties to the U.S.  
                    • Prior boarding foils or parole docs (if any)  
                    • Evidence of ongoing employment, family, or property in the U.S.  

                    **Step 2**: Use the following structure (raw Markdown, no code fences):

                    ```
                    #Demand for Adjudication Under the Mandamus Act and Administrative Procedure Act – [Applicant’s Full Name]  
                    **[Attorney’s Name]**  
                    **[Law Firm Name]**  
                    **[Street Address]**  
                    **[City, State, ZIP Code]**  
                    **[Phone Number]**  
                    **[Email Address]**  

                    **Date:** _YYYY-MM-DD_  

                    **RE:** _Demand for Adjudication under the Mandamus Act – Reentry Permit (Form I‑131) for [Applicant’s Full Name]_  

                    ### Dear USCIS Officer,  

                    **Introduction & Jurisdiction**  
                    - **Parties:** “This letter is submitted by **[Law Firm Name]** on behalf of **[Applicant’s Full Name]** (the “Applicant”), in support of the Reentry Permit (Form I‑131) application filed on [I‑131 Filing Date].”  
                    - **Procedural History:**  
                    - I‑131 Filed: _[Date]_; Receipt No.: _[Number]_  
                    - RFE Issued (if any): _[Date]_ → Response Filed: _[Date]_  
                    - Current Delay: _[Number]_ days beyond USCIS’s published 90‑day processing guideline  
                    - **Jurisdiction:** Demand is made under *28 U.S.C. § 1361* (mandamus) and *5 U.S.C. § 555(b)* (unreasonable delay).  

                    **Factual Background**  
                    - **Applicant Status:** Lawful Permanent Resident, A‑Number [#], PR card valid through [Date].  
                    - **Purpose of Travel:** [Describe reason—e.g., family emergency, employment abroad, education].  
                    - **Intended Absence:** Approximately [# months/years], departing [Departure Date], returning by [Expected Return Date].  
                    - **Procedural Compliance:** All required forms and fees submitted in accordance with USCIS guidelines.  

                    **Legal Standard for Mandamus**  
                    - **Clear Right:** Applicant’s undisputed right to timely adjudication of Form I‑131.  
                    - **Non‑Discretionary Duty:** USCIS must process travel‑document applications within a reasonable period.  
                    - **No Adequate Alternative:** Inquiries and service requests have not yielded a decision.  
                    - **Agency Guidelines:** USCIS processing goal is 90 days for Form I‑131 (see USCIS Processing Times webpage).  

                    **Demand for Relief**  
                    - **Relief Sought:** Final adjudication of the Reentry Permit application within **14 days** of receipt.  
                    - **Statutory Authority:** *28 U.S.C. § 1361*; *5 U.S.C. § 555(b)*.  
                    - **Consequences if Unresolved:**  
                    - **Applicant Hardship:** Risk of abandonment of residency, inability to reenter U.S. after [Departure Date].  
                    - **Family & Employment Impact:** Disruption of employment, education of dependents, and personal obligations.  

                    **Prejudice & Hardship**  
                    - **Applicant Impact:**  
                    - Potential loss of LPR status if permit not issued prior to departure.  
                    - Emotional and financial strain on family left in the U.S.  
                    - **Irreparable Injury:** Monetary damages inadequate; only mandamus relief will preserve Applicant’s right to reentry.  

                    **Exhibits & Supporting Documents**  
                    - **Ex. A:** Complete Form I‑131 petition package  
                    - **Ex. B:** USCIS receipt notice for Form I‑131  
                    - **Ex. C:** Copy of Permanent Resident Card  
                    - **Ex. D:** Two passport‑style photos  
                    - **Ex. E:** Applicant’s letter explaining purpose and duration of travel  
                    - **Ex. F:** Copy of Form I‑94  
                    - **Ex. G:** Passport biographic and visa pages  

                    **Conclusion & Next Steps**  
                    - **Final Demand:** “We request USCIS issue a final decision on Form I‑131 no later than 14 days from service of this demand.”  
                    - **Service Confirmation:** “Please confirm receipt via email to **[Attorney’s Email]** or fax to **[Fax Number]**.”  
                    - **Litigation Warning:** “Absent timely action, we will file a Writ of Mandamus in the U.S. District Court for the District of **[District]**, and seek appropriate fees and costs under the Equal Access to Justice Act.”  
                    - **Attorney Availability:** “[Attorney’s Name] is available to provide any further information or documentation USCIS may require.”  

                    **Sincerely,**  
                    \_\_\_\_\_\_\_\_\_\_\_  
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
                You are tasked with generating a Assessment Report in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Reentry Permit Support Letter – [Applicant’s Full Name]
                **[Consulate/USCIS Lockbox Name]**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Support Letter for Reentry Permit (Form I‑131) – [Applicant’s Full Name]  

                **Dear Consular Officer/USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”), hereby submits this support letter in conjunction with Form I‑131, Application for Travel Document, to secure a Reentry Permit under INA § 216(a).”  
                **Purpose of Travel:** To permit temporary departure from the United States for [reason for travel] and return within [intended length of absence].  

                **Background & Eligibility**  
                **Residency Status:** Applicant is a Lawful Permanent Resident (A‑Number [#]), holding a Permanent Resident Card valid through [date].  
                **Statutory Basis:** Requested under INA § 216(a) to maintain residence status while traveling abroad for [duration].  
                **Travel Details:** Departure on [departure date]; expected return by [return date].  

                **Required Forms & Evidence**  
                - **Form I‑131:** Completed and signed.  
                - **Form G‑1145:** E‑Notification of Application/Petition Acceptance.  
                - **Form G‑28:** Notice of Entry of Appearance as Attorney (if represented).  
                - **DS‑160:** Confirmation page (if consular processing required).  

                **Supporting Documents Provided**  
                1. Copy of Permanent Resident Card  
                2. Two passport‑style photos (USCIS specifications)  
                3. Copy of Form I‑94  
                4. Passport biographic and visa pages  
                5. Letter from Applicant explaining purpose and duration of travel  
                6. Proof of return‑ticket reservation or other evidence of U.S. ties 

                **Procedural Compliance**  
                All documentation complies with USCIS and Department of State requirements. The Applicant’s continuous residence in the United States is established, and no abandonment of status will occur.  

                **Conclusion & Request**  
                “Based on the foregoing, the Applicant respectfully requests prompt adjudication and issuance of the Reentry Permit. Please contact **[Attorney/Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]** for any questions or additional documentation.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_\_  
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  

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
        "Eligibility Memorandum": Agent(
            name="Eligibility Memorandum Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Eligibility Memorandum in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Reentry Permit Support Letter – [Applicant’s Full Name]
                **Consulate/USCIS Lockbox Name**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Eligibility Assessment Report for Reentry Permit (Form I‑131) – [Applicant’s Full Name]  

                **Dear Consular Officer/USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”) respectfully submits this Eligibility Assessment Report in support of the Reentry Permit (Form I‑131) application under INA § 216(a) to permit travel abroad for [intended length of absence] beginning [departure date].”  
                **Travel Overview:** Departure: _[Departure Date]_; Return: _[Expected Return Date]_; Purpose: _[Reason for Travel – e.g., family emergency, employment, education]_  

                **Statutory Basis**  
                Requested under INA § 216(a), which authorizes Lawful Permanent Residents to apply for a Reentry Permit to preserve continuous residency while abroad.  

                **Eligibility Evaluation**  
                - **Residency Status:** Applicant is a Lawful Permanent Resident, A‑Number [#], with Form I‑551 valid through [PR Card Expiration Date].  
                - **Non‑Abandonment of Residence:** Applicant maintains U.S. ties through ongoing employment ([Employer Name]), family ([Spouse/Children]), and property ownership ([Address or Description]).  
                - **Purpose & Duration:** Travel for [detailed purpose], with planned absence of approximately [# months/years], returning by [Expected Return Date].  
                - **Compliance with Requirements:** Applicant has not received prior removal proceedings or reentry refusals and has no known inadmissibility issues.  

                **Supporting Evidence & Exhibits**  
                **Exhibit A:** Copy of Permanent Resident Card (Form I‑551)  
                **Exhibit B:** Two passport‑style photos (USCIS specifications)  
                **Exhibit C:** Copy of Form I‑94 Arrival/Departure Record  
                **Exhibit D:** Passport biographic and visa pages  
                **Exhibit E:** Letter from Applicant explaining purpose and duration of travel  
                **Exhibit F:** Return‑ticket reservation or proof of ties to the U.S.
                **Exhibit G:** Prior boarding foils or parole documents.  
                **Exhibit H:** Evidence of ongoing U.S. ties (employment, family, property)

                **Procedural Compliance**  
                All forms and evidence have been compiled in strict accordance with USCIS and Department of State guidelines. The Applicant’s continuous U.S. residency is well‑documented, and all statutory and regulatory criteria have been satisfied.  

                **Conclusion & Request**  
                “Based on the foregoing analysis and supporting exhibits, the Applicant clearly meets the eligibility requirements for issuance of a Reentry Permit under INA § 216(a). We respectfully request prompt adjudication of this application. Please contact [Attorney/Representative Name] at [Phone Number] or [Email Address] for any further inquiries or documentation.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_\_ 
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  

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
                You are tasked with generating a Visa application in support of a Reentry Permit (Form I-131) application under Consular Processing.

                **Step 1**: Extract all required information *only* from the file(s) provided. Do not consult external sources or prior files. If any piece of information is missing, leave that placeholder blank—do not guess or invent data. Required data includes:
                - Applicant personal details (name, A‑number, date of birth, country of nationality)
                - Consulate/USCIS lockbox address and beneficiary’s mailing address
                - Travel dates, intended length of absence, and reason for travel
                - Attorney/representative details (if any)

                **Required Forms**  
                • Form I‑131, Application for Travel Document (USCIS)  
                • Form G‑1145, E‑Notification of Application/Petition Acceptance (USCIS)  
                • Form G‑28, Notice of Entry of Appearance as Attorney (USCIS)  
                • DS‑160, Online Nonimmigrant Visa Application (if applicable)  

                **Supporting Documents**  
                **Must Have**  
                • Copy of Permanent Resident Card (USCIS)  
                • Two passport‑style photos (USCIS photo specs)  
                • Form I‑94 copy (CBP)  
                • Passport biographic and visa pages (State/CBP)  
                • Explanation letter from applicant detailing purpose and length of intended absence  
                **As Available**  
                • Return‐ticket reservation or proof of strong ties to the U.S.  
                • Prior boarding foils or parole docs (if any)  
                • Evidence of ongoing employment, family, or property in the U.S.  

                **Step 2**: Use the following structure (raw Markdown, no code fences):
                ```
                #Reentry Permit Application Summary Report – [Applicant’s Full Name]
                **Consulate/USCIS Lockbox Name**  
                **[Address Line 1]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Reentry Permit Application Summary Report – [Applicant’s Full Name]  

                **Dear Consular Officer/USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Applicant’s Full Name] (the “Applicant”) submits this Summary Report in support of the Reentry Permit application (Form I‑131) under INA § 216(a).”  
                **Travel Overview:** Intended departure: _[Departure Date]_ from _[U.S. Port of Exit]_ and return by _[Expected Return Date]_.  
                **Objective:** Provide a concise narrative of the key evidence demonstrating the Applicant’s eligibility and continuity of Lawful Permanent Resident status.

                **Summary of Evidence**  
                The record clearly establishes that the Applicant maintains continuous U.S. residency and meets all statutory requirements for a Reentry Permit. A copy of the Permanent Resident Card valid through _[Date]_ confirms status continuity. Two passport‑style photos in compliance with USCIS specifications and the completed Form I‑131 demonstrate procedural correctness. The I‑94 arrival/departure record and biographic pages of the passport corroborate travel history. The letter from the Applicant detailing purpose (_[Reason for Travel]_) and intended absence of _[Length of Time]_ evidences bona fide intent to return. Supporting exhibits such as return‑ticket reservation and proof of U.S. ties (employment letter, property deeds) further substantiate non‑abandonment of residence.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing, the Applicant satisfies all requirements under INA § 216(a) for issuance of a Reentry Permit.”  
                **Request for Adjudication:** “Applicant respectfully requests prompt adjudication and approval of the Reentry Permit application.”  
                **Point of Contact:** “For any questions or additional documentation, please contact **[Attorney/Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_  
                **[Attorney/Representative Name], [Title]**  
                **[Law Firm/Company]**  

                ```

                Step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
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
