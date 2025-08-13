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
                You are tasked with generating a petition cover letter for an H-1B visa application.
                
                **Step 1**: Extract all necessary information from the vector store, including:
                - Personal details of the beneficiary.
                - Employer details.
                - Job description and duties.
                - Supporting evidence such as Form I-129, the H-Classification Supplement, certified LCA, and evidence of degree.

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
                Step 3.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 4.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 5.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 6.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
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
        "Eligibility Memorandum": Agent(
            name="Eligibility Memorandum Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating an employer support letter for an H-1B visa application.

                **Step 1**: Extract all necessary information from the vector store, including:
                - Employer details.
                - Job description and duties.
                - Relevance of the beneficiary’s degree to the job.
                - Salary details and justification.
                - Supporting evidence such as certified LCA and employer-provided letters.

                **Step 2**: Use the following structure for the letter:
                ```
                #Employer Support Letter for H-1B Visa Application – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Employer Support Letter for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) respectfully submits this Employer Support Letter on behalf of **[Beneficiary’s Name]** (the “Beneficiary”) in support of its H-1B petition.”  
                **Position Summary:** Title: _[Position Title]_; SOC Code: _[Code]_; Location(s): _[City, State]_.  
                **Statutory Framework:** Requested classification under INA § 214(i) for specialty occupations.  

                **Job Details**  
                **Duties & Responsibilities:**  
                    **Primary Duties:**  
                    1. _[Describe specialized analytical/design/programming duty requiring theoretical knowledge]_  
                    2. _[Describe project leadership or cross-functional coordination tasks]_  
                    **Secondary Duties:**  
                     _[List supportive tasks that nonetheless require degree-level skills]_  
                **Specialty Occupation Justification:**  
                    Demonstrates application of specialized knowledge in _[field]_ consistent with SOC Code _[Code]_ description.  
                    Duties require at least a bachelor’s degree in _[Field]_ (e.g., complex data modeling, software architecture).  

                **Degree Relevance**  
                **Beneficiary’s Academic Credentials:**  
                    _[Degree Type] in [Field]_ from _[University]_ (Date).  
                    Honors, thesis title, accreditation status.  
                **Direct Correlation to Duties:**  
                    Coursework in _[Key Subjects]_ equips Beneficiary to perform _[Duty #1]_.  
                    Specialized training in _[Tool/Method]_ essential for _[Duty #2]_.  
                **Policy Alignment:**  
                    Cite AAO decision or policy memo (e.g., “Matter of Michael Hertz Associates”) confirming degree-to-duty nexus.  

                **Salary and Need**  
                **Offered Salary:**  
                    Annual wage: _$[Amount]_ (at or above prevailing wage level _[Level]_ as per LCA certified _[Date]_).  
                    Overtime/bonus structure (if applicable).  
                **Business Justification:**  
                    Role critical for _[project/client name]_, directly impacting revenue of _$[Amount]_ per quarter.  
                    Lack of qualified U.S. applicants necessitates hiring the Beneficiary to meet specialized technical/manufacturing/service demands.  
                **Labor Condition Application (LCA) Compliance:**  
                    LCA certified by DOL on _[Date]_; public access file maintained per 20 C.F.R. § 655.760.  

                **Conclusion & Request**  
                **Eligibility Confirmation:** “Beneficiary’s specialized degree and experience clearly satisfy the requirements of a specialty occupation under INA § 214(i).”  
                **Request for Approval:** “Petitioner respectfully requests that USCIS approve the H-1B petition for **[Beneficiary’s Name]** at its earliest convenience.”  
                **Point of Contact:** “For any questions or additional information, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_, 
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [HR File]  

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
        "Evidence-Organization Chart": Agent(
            name="Evidence-Organization Chart Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating an LCA memorandum for an H-1B visa application.

                **Step 1**: Extract all necessary information from the vector store, including:
                - Certified LCA.
                - Prevailing wage documentation.
                - Public-access files.

                **Step 2**: Use the following structure for the memorandum:
                ```
                #Labor Condition Application (LCA) Memorandum for H-1B Visa Application – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** LCA Memorandum for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) submits this Labor Condition Application (LCA) Memorandum in support of its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                **Statutory Authority:** LCA filed under INA § 212(n) and 20 C.F.R. Part 655, Subpart H, to ensure compliance with prevailing wage and public disclosure requirements.  
                **Position Summary:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_.  

                **Prevailing Wage Determination**  
                 **Wage Source & Level:**  
                     Prevailing wage obtained from the Department of Labor’s Online Wage Library or independent wage survey (e.g., Bureau of Labor Statistics OES data) for SOC Code _[Code]_ in _[Metropolitan Area]_.  
                     Selected Wage Level: _[Level]_ (per 20 C.F.R. § 655.731).  
                 **Calculation Methodology:**  
                     Cross‐referenced DOL’s Occupational Employment Statistics with proprietary survey data to confirm median wage.  
                     Adjusted for geographic differential using BLS locality pay percentages.  
                 **Supporting Documents:**  
                     **Exhibit A:** DOL Prevailing Wage Determination notice (PWD Case No. _[Number]_) dated _[Date]_.  
                     **Exhibit B:** Copy of proprietary wage survey report (with methodology summary).  
                     **Exhibit C:** LCA certified by the Department of Labor on _[Date]_ (Case No. _[Number]_).  

                **Public-Disclosure Compliance**  
                  **Posting Requirements:**  
                     Posted LCA notice at the worksite and alternative worksites for 10 consecutive business days as required by 20 C.F.R. § 655.734.  
                     Notices included job title, SOC code, wage rate, and contact information for the Employer’s Representative.  
                **Internal Notice:**  
                    Provided notice to bargaining representative (if any) or, in absence thereof, posted notice where other employees in similar occupations are employed.  
                **Public Access File:**  
                    Maintained at the worksite, available for public inspection, containing:  
                    1. Certified LCA and supporting PWD.  
                    2. Documentation of wage postings and internal notices.  
                    3. Beneficiary’s wage rate and period of employment.  
                    4. Evidence of business necessity for alternate worksites (if applicable).  
                    **Exhibit D:** Index to Public Access File contents with dates and locations of postings.  

                **Conclusion & Certification**  
                **Compliance Statement:** “Petitioner confirms full compliance with all LCA requirements under INA § 212(n) and 20 C.F.R. Part 655, Subpart H, including accurate prevailing wage determination and public disclosure.”  
                **Record Retention:** Employer will retain the Public Access File for at least one year beyond the period of employment as required by 20 C.F.R. § 655.760.  
                **Point of Contact:** “For any questions or review of the Public Access File, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_, 
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [HR Compliance File], [Public Access File Index]  
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
                You are tasked with generating a response brief to rebut USCIS concerns regarding specialty occupation or degree equivalency.

                **Step 1**: Extract all necessary information from the vector store, including:
                - RFE details.
                - Personal information of the beneficiary.
                - Supporting evidence such as RFE notices, evidence of degree, employer letters, and certified LCA.

                **Step 2**: Use the following structure for the response brief:
                ```
                #Response to USCIS RFE – H-1B Petition for [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Response to Request for Evidence (RFE) – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction & RFE Reference**  
                **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) submits this Response to the Request for Evidence (RFE) issued in connection with its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                 **RFE Details:**  
                    Receipt Number: _[Number]_  
                    RFE Issued: _[Date]_  
                    Response Deadline: _[Date]_  
                 **Summary of USCIS Concerns:**  
                    1. Question as to whether the position qualifies as a specialty occupation.  
                    2. Sufficiency of evidence regarding Beneficiary’s degree and experience.  
                    3. Adequacy of the job duties description.  

                **Rebuttal to USCIS Concerns**  
                 **Concern 1: Specialty Occupation Qualification**  
                    **USCIS Position:** “Position duties appear routine and not requiring a bachelor’s degree in a specific specialty.”  
                    **Rebuttal:**  
                    Detailed breakdown of primary duties demonstrating theoretical and practical application of a specialized field (see Section 2.1).  
                    Citation of SOC Code _[Code]_ description and AAO precedent (e.g., *Matter of Michael Hertz Associates*).  
                 **Concern 2: Beneficiary’s Credentials**  
                    **USCIS Position:** “Insufficient evidence that Beneficiary holds a degree in the required specialty.”  
                    **Rebuttal:**  
                    Submitted certified copy of diploma and official transcript (Ex. A).  
                    Credential evaluation report confirming U.S. equivalency to a bachelor’s degree in _[Field]_ (Ex. B).  
                 **Concern 3: Job Duties Description**  
                    **USCIS Position:** “Job description lacks specificity regarding specialized tasks.”  
                    **Rebuttal:**  
                    Revised job description with granular tasks tied to degree-level knowledge (Ex. C).  
                    Organizational chart illustrating Beneficiary’s role and reporting relationships (Ex. D).  
                 **Additional Legal Authority:**  
                    Cite INA § 214(i) and 8 C.F.R. § 214.2(h)(4)(iii)(A) regarding specialty occupation definitions.  

                **Supporting Evidence & Exhibits**  
                 **Exhibit A:** Beneficiary’s diploma and official transcript (with certified translation).  
                 **Exhibit B:** Credential evaluation report by [Evaluator Name], dated _[Date]_.  
                 **Exhibit C:** Augmented job description, including task-level breakdown.  
                 **Exhibit D:** Organizational chart showing Beneficiary’s position within the team.  
                 **Exhibit E:** Prior H-1B approval notices (if applicable), with receipt numbers.  
                 **Exhibit F:** Letters of support from project managers detailing specialized duties performed.  

                **Conclusion & Request**  
                 **Eligibility Reaffirmed:** “Based on the expanded evidence and legal authorities cited, the Beneficiary clearly meets all requirements for classification in a specialty occupation under INA § 214(i).”  
                 **Request for Adjudication:** “Petitioner respectfully requests that USCIS approve the H-1B petition for **[Beneficiary’s Name]** promptly and notify the Petitioner by email at **[Email Address]**.”  
                 **Point of Contact:** “For any further questions or documentation requests, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [Immigration File]
                ```

                Step 3.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 4.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 5.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 6.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
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
    }

#Initialize agents with the given vector store ID function
async def generate_document(file_type, agents):
    """
    Generate a single document using the corresponding agent.
    
    Args:
        file_type (str): The type of document to generate (e.g., "petition_cover_letter").
        agents (dict): Dictionary of initialized agents.
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
