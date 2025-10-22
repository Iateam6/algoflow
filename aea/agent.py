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
                You are tasked with generating a petition cover letter for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Application for Change of Status (Form I-539) and Application for Employment Authorization Document (Form I-765)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **Principal Applicant:** [Insert Full Name of Principal Applicant]
                
                Dear Sir/Madam:
                
                Please accept the enclosed I-539 application to change the status of [Applicant's Full Name] to [Dependent Status] dependent status. [Applicant's] current [Status] will expire on [MM/DD/YYYY].
                
                The following documents are enclosed in support of this application:
                
                1. Two separate checks in the amount of:  
                   - $[Amount] for the Form I-539 application filing fee  
                   - $[Amount] for the Form I-765 application filing fee  
                2. Form G-1145, *e-Notification of Application/Petition Acceptance*  
                3. Form G-28, *Notice of Entry of Appearance as Attorney or Accredited Representative*  
                4. Form I-539, *Application to Change/Extend Nonimmigrant Status*  
                5. Form I-765, *Application for Employment Authorization*, with two passport-style photos attached  
                6. Documentation for the principal applicant ([Spouse's Full Name]):  
                   - Form I-797A Approval Notice for Form I-539, Petition for a Nonimmigrant Worker  
                   - Passport biographic page  
                7. Documentation for the dependent applicant ([Applicant's Full Name]):  
                   - Form I-94 and CBP travel history  
                   - F-1 Visa  
                   - Form I-20  
                   - Passport biographic page  
                   - Marriage certificate  
                
                We respectfully request the prompt adjudication of this application. Thank you for your attention to this matter.

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
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
                    max_num_results=5,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Eligibility Memorandum": Agent(
            name="Eligibility Memorandum Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Eligibility Memorandum for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Eligibility Memorandum in Support of Application for Employment Authorization (Form I-765)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **Eligibility Category:** [Insert Category, e.g., (c)(9), (c)(3)(C), etc.]  
                
                Dear Sir/Madam:  
                
                Please find enclosed this Eligibility Memorandum in support of [Applicant’s Full Name]’s Form I-765, *Application for Employment Authorization*. This memorandum outlines the applicable eligibility category, summarizes the factual background, and identifies the legal basis for the requested employment authorization.  
                
                **Background**  
                [Provide a concise description of the applicant’s current immigration status, class of admission, and any pending or approved applications/petitions related to the eligibility category. Include receipt numbers if available and only if present in the provided file.]  
                
                **Eligibility Basis**  
                [State the exact eligibility category (e.g., (c)(9) – pending adjustment of status; (c)(3)(C) – post-completion OPT; (c)(8) – asylum applicant, etc.) and briefly explain how the evidence enclosed meets the regulatory requirements under 8 CFR 274a.12.]  
                
                **Enclosures in Support of This Memorandum**  
                The following documents are provided in support of this memorandum and the corresponding Form I-765:  
                
                1. Form I-765, *Application for Employment Authorization*  
                2. Form G-1145, *E-Notification of Application/Petition Acceptance* (if provided)  
                3. Form G-28, *Notice of Entry of Appearance as Attorney* (if represented)  
                4. Two passport-style photos (per USCIS specifications)  
                5. Form I-94, *Arrival/Departure Record*  
                6. Copy of passport biographic page and U.S. visa (if available)  
                7. Prior EAD cards (front and back), if applicable  
                8. Category-specific supporting evidence (e.g., Form I-485/I-589/I-20 with recommendation, marriage certificate, USCIS notices)  
                
                We respectfully submit this memorandum and the enclosed supporting documentation to establish [Applicant’s Full Name]’s eligibility for the requested employment authorization. We request favorable adjudication at your earliest convenience.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_
                **[Attorney/Representative Name], [Title]**  
                **[Firm/Organization Name]**  

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
                    max_num_results=5,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Evidence-Organization Chart": Agent(
            name="Evidence-Organization Chart Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Evidence-Organization Chart for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
                ```
                **RE: Evidence – Organization Chart in Support of Application for Employment Authorization (Form I-765)**  
                **Applicant:** [Insert Full Name of Applicant]  
                **Eligibility Category:** [Insert Category, e.g., (c)(9), (c)(3)(C), etc.]  
                
                Dear Sir/Madam:  
                
                Please find enclosed the organizational chart of [Company/Organization Name] in support of [Applicant’s Full Name]’s Form I-765, *Application for Employment Authorization*. This document is submitted to demonstrate the applicant’s position within the company, the reporting structure, and the relationship to other personnel relevant to the eligibility category asserted.  
                
                **Purpose of Submission**  
                The organizational chart provides a clear visual representation of:  
                - The applicant’s current or proposed role within the company.  
                - The chain of command and supervisory relationships.  
                - How the applicant’s position aligns with the company’s operational structure.  
                - The relationship between the applicant’s position and other departments or personnel relevant to the application.  
                
                **Enclosures in Support of This Evidence**  
                1. Company organizational chart identifying the applicant’s position.  
                2. Legend or key explaining titles, reporting lines, and departments (if not clearly marked on the chart).  
                3. Supporting documentation (if available in the provided file set), such as:  
                   - Job description corresponding to the position shown.  
                   - Employer letter confirming the role and reporting structure.  
                   - Any relevant USCIS notices, contracts, or internal HR records.  
                
                We respectfully submit this organizational chart and accompanying evidence to assist USCIS in understanding [Applicant’s Full Name]’s position and its relation to the eligibility requirements for employment authorization.  
                
                **very truly yours,** 
                \_\_\_\_\_\_\_\_\_\_\_ 
                **[Attorney/Representative Name], [Title]**  
                **[Firm/Organization Name]**  

                ```

                Step 3.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 4.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 5.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 6.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file .
                Step 7. Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),
            model="gpt-4.1",
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=5,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Exhibit List": Agent(
            name="Exhibit List Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for an Application for Employment Authorization (Form I-765).

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Required forms:
                  - Form I-765 (Application for Employment Authorization)
                  - Form G-28 (Notice of Entry of Appearance as Attorney), if represented
                  - Form G-1145 (E-Notification of Application/Petition Acceptance)
                  - Form I-912 (Request for Fee Waiver), if applicable
                  - Form G-1450 (Authorization for Credit Card Transactions), if applicable
                - Supporting documents (include only those present in the provided file set; leave blank otherwise):
                  - Two passport-style photos (per USCIS specifications)
                  - Passport (biographic page) and U.S. visa page (if available)
                  - Form I-94 (Arrival/Departure Record)
                  - Prior EAD cards (front and back), if any
                  - Government-issued photo ID (if no passport available)

                **Step 2**: Use the following structure for the letter:
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
            model_settings=ModelSettings(temperature=0.9),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=5,
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
