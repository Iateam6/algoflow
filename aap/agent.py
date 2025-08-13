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
                #Petition Cover Letter for H-1B Visa Application – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Petition Cover Letter for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) respectfully submits this cover letter in support of its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_  
                **Statutory Basis:** Requested under INA § 214(i) governing specialty occupations.  

                **Eligibility – Specialty Occupation**  
                **Degree Requirement:** Position requires at minimum a bachelor’s degree or higher in _[Field of Study]_; Beneficiary holds a _[Degree]_ in _[Field]_ from _[Institution]_ (Date).  
                **Specialized Duties:**  
                    Duty 1: _[Describe primary function]_  
                    Duty 2: _[Describe advanced responsibility requiring theoretical application]_  
                    Duty 3: _[List any supervisory or collaborative tasks]_  
                **Alignment with USCIS Policy:**  
                    Compare job duties to SOC description for code _[Code]_.  
                    Cite policy memo (e.g., “Matter of Michael Hertz Associates” or AAO precedent).  
                **Beneficiary’s Qualifications:**  
                    Prior H-1B (if any): Receipt No. _[Number]_, Approval Date _[Date]_.  
                    Professional certifications: _[List]_ demonstrating non-routine expertise.  

                **Regulatory & Procedural Compliance**  
                **Labor Condition Application (LCA):** Certified by Department of Labor on _[Date]_ for wage level _[Level]_ at worksite _[Address]_.  
                **Public Access File:** Documentation available at the worksite in compliance with 20 C.F.R. § 655.760.  
                **Maintenance of Status:** Beneficiary’s current status (_[e.g., F-1 OPT]_) valid through _[Date]_; no gap anticipated.  
                **Dependent Filings (if applicable):** H-4 petitions for _[Spouse/Children]_ filed concurrently (Receipt Nos. _[Numbers]_).  

                **Supporting Evidence & Exhibits**  
                **Exhibit A:** Beneficiary’s diploma(s) and transcripts (with certified translations).  
                **Exhibit B:** Detailed resume/CV and letters of employment verification.  
                **Exhibit C:** Signed job offer letter and comprehensive job description.  
                **Exhibit D:** Organizational chart showing Beneficiary’s role and reporting structure.  
                **Exhibit E:** Prevailing Wage Determination (PWD) or certified LCA.  
                **Exhibit F:** Professional licenses, patents, publications, or conference presentations.  

                **Legal Standard & Precedent**  
                **Specialty Occupation Test (INA § 214(i)):** Position requires theoretical and practical application of highly specialized knowledge.  
                **AAO Precedents:** Cite decisions where similar roles were approved (e.g., IT systems analyst, financial analyst).  
                **Burden of Proof:** Petitioner has met its burden to show Beneficiary’s qualifications and position requirements.  

                **Conclusion & Request for Favorable Adjudication**  
                “Based on the foregoing, the Petitioner has demonstrated that the Beneficiary qualifies for classification in a specialty occupation under INA § 214(i). Petitioner respectfully requests that USCIS approve this H-1B petition promptly.”  
                **Point of Contact:** For any questions or additional documentation, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [HR File]  
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
        "Exhibit List": Agent(
            name="Exhibit List Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a Exhibit List for an H-1B visa application.
                
                **Step 1**: Extract all necessary information from the vector store, including:
                - Personal details of the beneficiary.
                - Employer details.
                - Job description and duties.
                - Supporting evidence such as Form I-129, the H-Classification Supplement, certified LCA, and evidence of degree.

                **Step 2**: Use the following structure for the letter:
                ```
                #Petition Cover Letter for H-1B Visa Application – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Petition Cover Letter for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) respectfully submits this cover letter in support of its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_  
                **Statutory Basis:** Requested under INA § 214(i) governing specialty occupations.  

                **Eligibility – Specialty Occupation**  
                **Degree Requirement:** Position requires at minimum a bachelor’s degree or higher in _[Field of Study]_; Beneficiary holds a _[Degree]_ in _[Field]_ from _[Institution]_ (Date).  
                **Specialized Duties:**  
                    Duty 1: _[Describe primary function]_  
                    Duty 2: _[Describe advanced responsibility requiring theoretical application]_  
                    Duty 3: _[List any supervisory or collaborative tasks]_  
                **Alignment with USCIS Policy:**  
                    Compare job duties to SOC description for code _[Code]_.  
                    Cite policy memo (e.g., “Matter of Michael Hertz Associates” or AAO precedent).  
                **Beneficiary’s Qualifications:**  
                    Prior H-1B (if any): Receipt No. _[Number]_, Approval Date _[Date]_.  
                    Professional certifications: _[List]_ demonstrating non-routine expertise.  

                **Regulatory & Procedural Compliance**  
                **Labor Condition Application (LCA):** Certified by Department of Labor on _[Date]_ for wage level _[Level]_ at worksite _[Address]_.  
                **Public Access File:** Documentation available at the worksite in compliance with 20 C.F.R. § 655.760.  
                **Maintenance of Status:** Beneficiary’s current status (_[e.g., F-1 OPT]_) valid through _[Date]_; no gap anticipated.  
                **Dependent Filings (if applicable):** H-4 petitions for _[Spouse/Children]_ filed concurrently (Receipt Nos. _[Numbers]_).  

                **Supporting Evidence & Exhibits**  
                **Exhibit A:** Beneficiary’s diploma(s) and transcripts (with certified translations).  
                **Exhibit B:** Detailed resume/CV and letters of employment verification.  
                **Exhibit C:** Signed job offer letter and comprehensive job description.  
                **Exhibit D:** Organizational chart showing Beneficiary’s role and reporting structure.  
                **Exhibit E:** Prevailing Wage Determination (PWD) or certified LCA.  
                **Exhibit F:** Professional licenses, patents, publications, or conference presentations.  

                **Legal Standard & Precedent**  
                **Specialty Occupation Test (INA § 214(i)):** Position requires theoretical and practical application of highly specialized knowledge.  
                **AAO Precedents:** Cite decisions where similar roles were approved (e.g., IT systems analyst, financial analyst).  
                **Burden of Proof:** Petitioner has met its burden to show Beneficiary’s qualifications and position requirements.  

                **Conclusion & Request for Favorable Adjudication**  
                “Based on the foregoing, the Petitioner has demonstrated that the Beneficiary qualifies for classification in a specialty occupation under INA § 214(i). Petitioner respectfully requests that USCIS approve this H-1B petition promptly.”  
                **Point of Contact:** For any questions or additional documentation, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [HR File]  
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
        "Employer Support Letter": Agent(
            name="Employer Support Letter Agent",
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
        "LCA Memorandum": Agent(
            name="LCA Memorandum Agent",
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
        "RFE Response Brief": Agent(
            name="RFE Response Brief Agent",
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
        "Data-Collection Supplement Attachments": Agent(
            name="Data Collection Supplement Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a data-collection supplement for cases filed under cap-exempt or fee-exemption categories.

                **Step 1**: Extract all necessary information from the vector store, including:
                - Exemption details.
                - Supporting evidence such as the H-Classification Supplement, certified LCA, and exemption documents.

                **Step 2**: Use the following structure for the supplement:
                ```
                #H-1B Data Collection and Filing Fee Exemption Supplement – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Data Collection Supplement for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                - **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) submits this Data Collection Supplement in support of its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                - **Statutory Basis:** This supplement is provided pursuant to USCIS policy guidance on H-1B Data Collection and Filing Fee Exemptions (8 C.F.R. § 214.2(h)(19)).  
                - **Position Information:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_.  

                **Exemption Details**  
                 **Exemption Category:**  
                    ☐ Cap-exempt institution under INA § 214(g)(5)(C) (e.g., institution of higher education)  
                    ☐ Nonprofit affiliated with an institution of higher education under INA § 214(g)(5)(A)  
                    ☐ Government research organization under INA § 214(g)(5)(B)  
                    ☐ Fee-exempt nonprofit research organization under 8 C.F.R. § 214.2(h)(19)  
                **Justification for Exemption:**  
                    Describe organizational status (e.g., “Nonprofit research entity registered under Section 501(c)(3) of the Internal Revenue Code”).  
                    Explain affiliation or control relationship (e.g., memorandum of understanding with University X).  
                    Provide rationale: research focus, public benefit, no commercial gain.  
                **Regulatory Compliance:**  
                    Cite relevant USCIS policy memoranda (e.g., “Fee Exemption for Nonprofit Research Organizations,” HQ 70/6.5.2†).  
                    Confirm that placement of Beneficiary does not generate net profit for Petitioner.  

                **Supporting Evidence & Exhibits**  
                  **Exhibit A:** IRS determination letter evidencing 501(c)(3) status (Date: _[Date]_).  
                  **Exhibit B:** Articles of incorporation and bylaws showing nonprofit mission and governance.  
                  **Exhibit C:** Affiliation agreement or control documentation with institution of higher education (if applicable).  
                  **Exhibit D:** Organizational chart highlighting research units and Beneficiary’s position.  
                  **Exhibit E:** Form ETA-9035 data collection supplement completed and signed by Petitioner.  
                  **Exhibit F:** Job description emphasizing duties in furtherance of nonprofit research or education.  

                **Conclusion & Certification**  
                  **Compliance Statement:** “Petitioner certifies that all information provided is true and correct, and that the Beneficiary’s employment falls squarely within the claimed exemption category.”  
                  **Record Retention:** “All supporting documentation will be maintained in the Public Access File as required by 20 C.F.R. § 655.760.”  
                  **Request for Consideration:** “Petitioner respectfully requests that USCIS recognize the exemption and adjudicate the H-1B petition without cap or fee restrictions.”  
                  **Point of Contact:** “For any further inquiries or to review additional documentation, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [HR Compliance File]  

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
        "Demand Letter": Agent(
            name="Demand Letter Agent",
            instructions=(
                        f"""
                        Today’s date is {current_date}.  
                        You are an expert immigration attorney tasked with generating a formal demand letter to U.S. Citizenship and Immigration Services (USCIS) under the Mandamus Act, compelling them to adjudicate a pending H-1B petition.    
                        Step 1. Extract from the vector store all client & attorney details, employer justifications, certified LCA, prevailing-wage docs, RFE responses, and supporting evidence.  
                        Step 2. Use this exact structure (from GPT-Demand-1.docx) and format it in Markdown:
    
                        ```
                        #Demand for Adjudication Under the Mandamus Act and Administrative Procedure Act – [Beneficiary’s Full Name] 
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
                        - **Parties:** “This letter is submitted by **[Law Firm Name]** on behalf of **[Employer Name]** (the “Petitioner”) in support of its H-1B petition for **[Beneficiary Name]** (the “Beneficiary”).”  
                        - **Procedural History:**  
                          - I-129 Filed: _[Date]_; Receipt No.: _[Number]_  
                          - RFE Issued: _[Date]_ → Response Filed: _[Date]_  
                          - Current Delay: _[Number]_ days beyond USCIS’s 60-day guideline  
                        - **Jurisdiction:** Demand is made under *28 U.S.C. § 1361* (mandamus) and *5 U.S.C. § 555(b)* (unreasonable delay).  

                        **Factual Background**  
                        - **Employer Profile:** Industry, size, nature of business, and critical need for Beneficiary’s skills.  
                        - **Beneficiary Credentials:** Degree, field, years of experience, prior visa status.  
                        - **Position Details:** Title, SOC code, wage level, project description, worksite location(s).  
                        - **Key Dates (Timeline):**  
                          - • _[Date]_ – I-129 Filed  
                          - • _[Date]_ – RFE Issued (Ex. B)  
                          - • _[Date]_ – RFE Response Filed (Ex. C)  
                          - • _Today’s Date_ – Over _[X]_ days past target  

                        **Legal Standard for Mandamus**  
                        - **Clear Right:** Petitioner’s indisputable right to a decision.  
                        - **Non-Discretionary Duty:** USCIS must adjudicate within reasonable time.  
                        - **No Adequate Alternative:** Status inquiry or service request is insufficient.  
                        - **Agency Guidelines:** USCIS aims to resolve RFEs within 60 days (see July 17, 2017 Policy Memo).  

                        **Demand for Relief**  
                        - **Relief Sought:** Adjudication of the H-1B petition within **14 days** of receipt.  
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

                        **Exhibits & Supporting Documents**  
                        - **Ex. A:** Redacted I-129 petition package  
                        - **Ex. B:** USCIS receipt notices  
                        - **Ex. C:** RFE and response correspondence  
                        - **Ex. D:** Beneficiary’s CV, degree certificates  
                        - **Ex. E:** Client contract summary / org chart  

                        **Conclusion & Next Steps**  
                        - **Final Demand:** “We request USCIS issue a final decision no later than 14 days from service.”  
                        - **Service Confirmation:** “Please confirm receipt via email to **[Attorney’s Email]** or fax to **[Fax Number]**.”  
                        - **Litigation Warning:** “Absent timely action, we will file a Writ of Mandamus in the U.S. District Court for the District of **[District]**, and seek EAJA fees and costs.”  
                        - **Attorney Availability:** “[Attorney Name] is available to provide any further information USCIS may require.”  

                        **Sincerely,** 
                        \_\_\_\_\_\_\_\_\_\_\_,
                        **[Attorney’s Full Name], Esq.**  
                        **[Law Firm Name]**  

                        **cc:** [Employer HR / In-House Counsel], [Other Relevant Parties] 

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
        "Assessment Report": Agent(
            name="Assessment Report Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating an assessment report summarizing the applicant's qualifications and eligibility for an H-1B visa.

                **Step 1**: Extract all necessary information from the vector store, including:
                - Personal details of the beneficiary.
                - Employer details.
                - Supporting evidence such as evidence of degree, certified LCA, and passport/I-94 copies.

                **Step 2**: Use the following structure for the report:
                ```
                #H-1B Specialty Occupation Support Letter – Electrical Engineer Position for [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Assessment Report for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                **Parties:** “[Employer’s Name] (the “Petitioner”) submits this Assessment Report in support of its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                **Purpose:** To provide a comprehensive evaluation of the Beneficiary’s credentials and demonstrate statutory eligibility under INA § 214(i).  
                **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Wage Level: _[Level]_; Location(s): _[City, State]_.

                **Employer & Position Description**  
                **Employer Profile:**  
                Industry, size, years in operation.  
                Core business activities and key clients.  
                Why specialized expertise of the Beneficiary is essential to operations/projects.  
                **Job Duties & Requirements:**  
                Detailed list of primary and ancillary duties.  
                Minimum education and experience prerequisites.  
                Specialized tools, methodologies, software, or processes required.

                **Summary of Qualifications**  
                **Educational Background:**  
                    Degree(s) earned (e.g., B.S., M.S., Ph.D.) in _[Field]_ from _[Institution]_ (Date).  
                    Honors, thesis title, accredited status of institution.  
                **Professional Experience:**  
                    _[Years]_ years at _[Company]_ as _[Role]_; key achievements and project summaries.  
                    Prior H-1B or other visa status (if applicable) with USCIS receipt numbers and approval dates.  
                **Specialized Knowledge & Skills:**  
                    Technical proficiencies (software, programming languages, analytical techniques).  
                    Certifications, published papers, patents, or speaking engagements.  
                    Unique contributions to past or ongoing projects demonstrating non-routine expertise.

                **Alignment with Regulatory Criteria**  
                **“Specialty Occupation” Analysis (INA § 214(i)):**  
                    Explain how the position requires a bachelor’s (or higher) degree in a specific specialty.  
                    Compare job duties to standard SOC descriptions and educational prerequisites.  
                **“Beneficiary’s Qualifications” Analysis:**  
                    Connect each degree and experience bullet to a corresponding duty or requirement.  
                    Cite USCIS policy memoranda or AAO decisions where similar profiles were approved.

                **Supporting Evidence & Exhibits**  
                **Exhibit A:** Copy of Beneficiary’s diploma(s) and transcripts (with English translations).  
                **Exhibit B:** Detailed resume/CV, employment verification letters, and reference contacts.  
                **Exhibit C:** Job offer letter, detailed job description, and organizational chart.  
                **Exhibit D:** Professional certifications, patents, publications, or awards.  
                **Exhibit E:** Comparative wage data (Prevailing Wage Determination or LCA).

                **Legal & Procedural Compliance**  
                **Labor Condition Application (LCA):** LCA certified on _[Date]_; wage level and worksite locations match.  
                **Public Access File:** Confirm availability of required documentation at worksite.  
                **Dependents & Maintenance of Status:** Brief note on any accompanying H-4 or E-dependent filings.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the foregoing, the Beneficiary clearly meets the educational and experiential requirements for the specialty occupation.”  
                **Favorable Adjudication Sought:** “Petitioner respectfully requests that USCIS approve the H-1B petition for **[Beneficiary’s Name]** promptly, in accordance with INA § 214(i).”  
                **Point of Contact:** “Please direct any questions or requests for additional information to **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name, if applicable], [Beneficiary], [HR File] 
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
        "Eligibility Assessment Report": Agent(
            name="Eligibility Assessment Report Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating an eligibility assessment report evaluating the applicant's qualifications against H-1B visa requirements.

                **Step 1**: Extract all necessary information from the vector store, including:
                - Personal details of the beneficiary.
                - Employer details.
                - Supporting evidence such as Form I-129, evidence of degree, and employer letters.

                **Step 2**: Use the following structure for the report:
                ```
                #H-1B Specialty Occupation Support Letter – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Eligibility Assessment Report for H-1B Visa Application – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                  **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) respectfully submits this Eligibility Assessment Report on behalf of **[Beneficiary’s Name]** (the “Beneficiary”) in support of its H-1B petition.”  
                  **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_.  
                  **Statutory Basis:** Assessment conducted under INA § 214(i) governing specialty occupations.  

                **Eligibility Evaluation**  
                  **Specialty Occupation Criteria (INA § 214(i) & 8 C.F.R. § 214.2(h)(4)(iii)):**  
                      Position requires theoretical and practical application of a body of highly specialized knowledge.  
                      Duties are sufficiently complex and specialized to require at least a bachelor’s degree in _[Field]_.  
                **Degree-to-Job Relevance:**  
                      Beneficiary holds a _[Degree Type]_ in _[Field]_ from _[Institution]_ (Date), directly aligning with the position’s academic requirements.  
                      Coursework in _[Key Subjects]_ equips Beneficiary to perform _[Duty 1]_ and _[Duty 2]_.  
                **Employer’s Business Need:**  
                      Role critical to _[Project/Client]_ with anticipated revenue impact of _$[Amount]_ per quarter.  
                      No qualified U.S. applicant possesses equivalent combination of education and specialized experience.  
                      Vacancy would materially delay deliverables and harm competitive standing.  

                **Supporting Evidence & Exhibits**  
                  **Exhibit A:** Certified copy of Beneficiary’s diploma and official transcripts (with translations).  
                  **Exhibit B:** Detailed resume/CV and letters of prior employment verification.  
                  **Exhibit C:** Comprehensive job description detailing specialized duties and SOC alignment.  
                  **Exhibit D:** Prevailing Wage Determination or certified LCA (Case No. _[Number]_) dated _[Date]_.  
                  **Exhibit E:** Organizational chart and project plan demonstrating critical role of Beneficiary.  

                **Conclusion & Request**  
                  **Eligibility Reaffirmed:** “Based on the foregoing analysis and supporting evidence, the Beneficiary meets all requirements for classification in a specialty occupation under INA § 214(i).”  
                  **Request for Adjudication:** “Petitioner respectfully requests that USCIS approve this H-1B petition for **[Beneficiary’s Name]** promptly.”  
                  **Point of Contact:** “For any questions or additional information, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [HR Compliance File]  
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
        "Visa Application Summary Report": Agent(
            name="Visa Application Summary Report Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a visa application summary report consolidating all relevant information for submission.

                **Step 1**: Extract all necessary information from the vector store, including:
                - Personal details of the beneficiary.
                - Employer details.
                - Supporting evidence such as Form I-129, certified LCA, evidence of degree, and employer letters.

                **Step 2**: Use the following structure for the summary report:
                ```
                #H-1B Visa Application Summary Report – [Beneficiary’s Full Name]
                **[Employer’s Name]**  
                **[Employer’s Address]**  
                **[City, State, ZIP Code]**  

                **Date:** _YYYY-MM-DD_  

                **Subject:** Visa Application Summary Report – [Beneficiary’s Full Name]  

                **Dear USCIS Officer,**  

                **Introduction**  
                  **Parties & Purpose:** “[Employer’s Name] (the “Petitioner”) submits this Visa Application Summary Report in support of its H-1B petition for **[Beneficiary’s Name]** (the “Beneficiary”).”  
                  **Position Overview:** Title: _[Position Title]_; SOC Code: _[Code]_; Worksite: _[City, State]_.  
                  **Objective:** Provide a concise narrative of the key evidence establishing the Beneficiary’s eligibility under INA § 214(i).  

                **Summary of Evidence**  
                The record demonstrates beyond question that the Beneficiary possesses the required academic credentials, specialized experience, and employer need for the role. The certified diploma and official transcripts from _[Institution]_ establish the Bachelor’s/Master’s degree in _[Field]_ directly related to the position’s theoretical and practical demands. Detailed employment verification letters and the resume illustrate _[X]_ years of progressive responsibility in _[Specialty Area]_, including leadership of complex projects and demonstrated proficiency with _[Key Tools/Technologies]_. The Labor Condition Application, certified on _[Date]_, confirms the prevailing wage compliance and worksite details. The comprehensive job description and organizational chart contextualize the Beneficiary’s unique contributions to critical client deliverables, underscoring the absence of comparably qualified U.S. applicants. Finally, credential evaluation reports and precedent-citing memoranda reinforce that the duties meet the regulatory definition of a specialty occupation.  

                **Conclusion & Request**  
                  **Eligibility Reaffirmed:** “Based on the foregoing, the Beneficiary incontrovertibly satisfies all statutory and regulatory requirements for H-1B classification.”  
                  **Request for Adjudication:** “Petitioner respectfully requests that USCIS approve the H-1B petition for **[Beneficiary’s Name]** without delay.”  
                  **Point of Contact:** “For any questions or additional documentation, please contact **[Employer’s Representative Name]**, **[Title]**, at **[Phone Number]** or **[Email Address]**.”  

                **Sincerely,**  
                \_\_\_\_\_\_\_\_\_\_\_,
                **[Employer’s Representative Name], [Title]**  
                **[Company Name]**  

                **cc:** [Attorney’s Name (if applicable)], [Beneficiary], [Immigration File]  

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
