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
                You are tasked with generating a petition cover letter for Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ``` 
                **RE: Application for Reentry Permit (Form I-131)**  
                **Applicant:** [Insert Full Name]  
                **Alien Number:** [Insert A-Number]  
                
                Dear Sir/Madam,
                
                We are submitting the following documents in support of the above applicant’s **Application for Reentry Permit (Form I-131)**:
                
                1 Check in the amount of $[Insert Amount] to cover the **Form I-131** filing fee and **biometric services**  
                2 **Form G-1145**, *e-Notification of Application/Petition Acceptance*  
                3 **Form G-28**, *Notice of Entry of Appearance as Attorney or Accredited Representative*  
                4 **Form I-131**, *Application for Travel Document*  
                5 Copy of the **front and back** of the applicant’s **green card**  
                6 Copy of the applicant’s **passport biographic page**  
                
                We appreciate your prompt attention to this matter.


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
        "Exhibit List": Agent(
            name="Exhibit List Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ``` 
                                                    Exhibit List  
                                    Self-Petitioner: [Beneficiary’s Full Name]  
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
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Eligibility-Checklist Memo": Agent(
            name="Eligibility-Checklist Memo Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating an Eligibility-Checklist Memo for Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ```
                # Eligibility‑Checklist Memo – Naturalization (Form N‑400)  
                **[Applicant’s Full Name]**  
                **[Applicant’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** Naturalization Eligibility Checklist Memo – [Applicant’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Purpose:** “[Applicant’s Full Name] submits this Eligibility‑Checklist Memo to accompany their Form N‑400, Application for Naturalization, summarizing key eligibility requirements and supporting evidence.”  
                **Basis of Filing:** “Applicant files under [General Provision: 5‑year permanent resident; Married to U.S. Citizen: 3‑year rule; Military service; VAWA, etc.] per INA § [section].”  

                **Eligibility Checklist**  
                Below is a narrative summary confirming that the Applicant meets each naturalization requirement under INA, supported by the evidence in the filing package:

                - **18 years of age or older**: Applicant is **[age]**, born on **[DOB]** (≥ 18 years at filing).  
                - **Lawful Permanent Resident status**: Green Card issued on **[Date]**, Alien Registration Number: **[A‑Number]**.  
                - **Continuous residence requirement**: 
                - Under general eligibility: permanent resident for at least 5 years (residency began on **[Date]**).
                - If eligible via marriage: married and living with U.S. citizen spouse since **[Date]**, meeting the 3‑year rule.  
                - **Physical presence requirement**: Applicant has been physically present in the U.S. for ≥ 30 months during the past 5 years (or ≥ 18 months in past 3 if under 3-year rule), as documented in travel history worksheets.  
                - **Residency in USCIS district / state**: Applicant has lived in **[State/District]** continuously since **[Date]**, satisfying the 3‑month residency requirement.  
                - **English language ability and civics knowledge**: Applicant can read, write, and speak basic English and demonstrates knowledge of U.S. history and government, supported by eligibility under standard testing or applicable exemptions.  
                - **Good moral character**: Applicant has no disqualifying criminal history; any arrests or convictions are detailed with documentation, and eligibility is preserved under relevant INA provisions.  
                - **Attachment to the Constitution**: Applicant affirms willingness to support the Constitution of the United States and take the Oath of Allegiance.  

                **Supporting Evidence**  
                The application packet includes the following supporting documentation:  
                - Copy of both sides of the Permanent Resident Card (Form I‑551)  
                - Travel history record summarizing trips outside the U.S.  
                - Residency and employment history covering the required period  
                - English and civics test results or applicable test waiver documentation (e.g. Form N‑648)  
                - Arrest records, court dispositions, or explanations regarding any criminal record (if applicable)  
                - Signed declaration affirming attachment to the Constitution and willingness to take the Oath  

                **Conclusion & Request**  
                **Eligibility Confirmed:** “Based on the above facts and enclosed evidence, the Applicant clearly satisfies all statutory and regulatory requirements for naturalization under INA eligibility criteria.”  
                **Request for Adjudication:** “Applicant respectfully requests that USCIS accept and adjudicate their N‑400 application and schedule a naturalization interview and oath ceremony without delay.”  

                **Point of Contact:** For any questions or additional information, please contact **[Preparer or Representative Name]** at **[Phone Number]** or **[Email Address]**.  

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Preparer’s Name], [Title or Role]**  
                **[Organization, if applicable]**  
  
              
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
        "Good Moral Character Brief": Agent(
            name="Good Moral Character Brief Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating an Good Moral Character Brief for a Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ```
                # Good Moral Character Brief – [Applicant’s Full Name]  

                **Date:** _YYYY‑MM‑DD_  

                **Subject:** Good Moral Character Brief – Naturalization of **[Applicant’s Full Name]**  

                **Dear USCIS Officer,**  

                **Background & Statutory Requirement:**  
                Pursuant to INA § 316(e) and 8 CFR § 316.10, the Applicant must demonstrate good moral character (GMC) during the statutory period: the five years immediately preceding filing (or three years if filing based on marriage to a U.S. citizen), and continuing up to the date of the Oath of Allegiance.

                **Evidence of Positive Conduct:**  
                During the required period, the Applicant has maintained steady lawful permanent residence, paid all required federal and state taxes, and complied with community and civic responsibilities. There are no arrests, convictions, or citations during the GMC period or otherwise. The Applicant has demonstrated honesty in all immigration filings and interviews, including full and truthful responses to all questions on Form N‑400 and during prior immigration proceedings.

                **Relevant Supporting Documentation:**  
                - IRS tax transcripts and W‑2 statements for the GMC period confirming timely filing and payment of taxes  
                - Personal affidavits and letters of recommendation attesting to honesty, reliability, and contributions to the local community (e.g. volunteer service, family support)  
                - Evidence of Selective Service registration (if applicable), and no incidents of failure to register during the eligible period  

                **No Disqualifying Conduct:**  
                The Applicant has not committed any crimes involving moral turpitude (CIMTs), aggravated felonies, or unlawful acts that would automatically disqualify GMC under INA §101(f) or 8 CFR § 316.10(b). There have been no false claims to U.S. citizenship, no immigration fraud, and no misuse of public benefits that would reflect dishonesty or poor character.

                **Totality of Circumstances & Character Rehabilitation (if applicable):**  
                There is no adverse conduct outside the statutory period that bears negatively on the Applicant’s present moral character. Even if past incidents existed, they would be outside the GMC period, and the Applicant’s subsequent behavior demonstrates full reformation and alignment with U.S. standards of good moral character.

                **Conclusion & Request:**  
                Based on the foregoing and the attached documentation, the Applicant clearly meets and continues to meet the statutory and regulatory requirements for **Good Moral Character**. The Applicant respectfully requests favorable consideration of these factors in adjudicating the N‑400 naturalization application.

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Applicant’s Name]**


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
                You are tasked with generating an Good Moral Character Brief for a Naturalization (N-400) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                    - N-400
                    - G-28
                    - G-1145
                - Supporting documents:
                   - Passport
                   - Visa pages
                   - Travel itinerary
                  
                **Step 2**: Use the following structure bellow:
                ```
                # RE: NOTICE OF CONTINUANCE (FORM N-400)

                **Applicant:** [Full Name of Applicant]  
                **Alien Number (A#):** [A-Number]
                
                Dear Sir or Madam,
                
                The United States Citizenship and Immigration Services (USCIS) issued a **Notice of Continuance** for the above-referenced case on **[Date of Original Notice]**. This response is being submitted pursuant to the extended timelines allowed by USCIS based on its flexibility policy announced on **[Flexibility Announcement Date]**.
                
                USCIS allows a response to be submitted within 60 calendar days after the deadline set forth in the original Notice of Continuance, provided the issuance date on the notice falls between **[Start Date]** and **[End Date]**, inclusive. The original notice in this case was issued on **[Date of Issuance]**, and therefore this response is timely.
                
                ## Reason for Continuance:
                
                USCIS records indicate that the applicant traveled outside the United States for a period of six months or more following lawful admission for permanent residence. As such, additional documentation is required to demonstrate that the applicant **did not abandon U.S. residency**.
                
                ## Enclosed Documentation:
                
                1 Copy of [Year] Tax Return  
                2 Notice of Immigrant Visa Case Creation  
                3 Affidavit from [Name or Relationship] affirming that [Affiant] maintained the applicant’s property during the applicant’s absence from the U.S.  
                4 Affidavit from [Name or Relationship] affirming that [Affiant] maintained the applicant’s property during the applicant’s absence from the U.S.
                
                Thank you for your favorable consideration of the enclosed documents. Should additional information be needed to make a decision on this case, please do not hesitate to contact us.

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Applicant’s Name]**


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
