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
        "DS-160 Completion Guide": Agent(
            name="DS-160 Completion Guide Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a DS-160 Interview Cheat-Sheet for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page

                **Step 2**: Use the following structure for the letter:
                ``` 
                ## DS‑160 Completion Guide

                **Passport Details**  
                Ensure the applicant provides a valid passport biographical page, bearing full name, date of birth, passport number, country of issuance, and expiration date. The passport must remain valid for at least six months beyond the intended U.S. stay (unless exempt by bilateral agreement). Accurate transcription into the DS‑160 is critical to avoid matching errors during reciprocity or entry. If any passport fields are missing in the source file (e.g. issue date or controlling authority), leave those entries blank rather than guessing.

                **Travel Itinerary**  
                If travel dates, intended U.S. departure and return plans, cities, or purpose of visit are provided, summarize them clearly. Include flight details if available. If no itinerary was included in the DS‑160 file, leave this section blank—do not assume destination or duration. Accurate itinerary detail helps demonstrate intent of temporary stay and supports the consular understanding of trip planning.

                **Previous U.S. Travel History**  
                List any prior visits to the U.S., including dates of entry and exit, visa types used, or prior refusals or overstays if documented. This section helps consular officers assess patterns of compliance or risk. If the DS‑160-derived file lacks travel history information, omit this section entirely to avoid imprinting unverified values and triggering inconsistencies.

                **Education and Employment History**  
                Provide details of the applicant’s education and employment over the past few years as captured in DS‑160: institutions attended, degrees earned, current and former employers, job titles, and dates. These details support evaluation of eligibility for visa categories like F, J, M, work, or business types. If fields are omitted in the source, leave them blank.

                **Social Media Usernames**  
                As of mid‑2025, applicants—including those applying for F, M, and J visas—must list **all** usernames or handles used on social media platforms in the past five years (e.g. Facebook, Instagram, LinkedIn, YouTube, X, Reddit, Tumblr, etc.). If applicants used platforms in the past five years, these identifiers must be entered even if accounts are inactive or deleted. Set these accounts to public during application review if required. If no platforms were used, the applicant may enter "None."

                **Digital Photo Requirements**  
                Confirm that a compliant photo file was uploaded with the DS‑160. The photo must meet U.S. visa specs: white background, 2×2 inches, head size between 50–69% of frame, neutral expression. If the original file lacks photo metadata, leave this placeholder blank and note that a valid digital photo must still be provided before submission.

                **DS‑160 Confirmation Page**  
                Ensure extraction of the confirmation barcode page details: application ID number, barcode, and date of submission. This page is required during the visa interview. If missing in the provided DS‑160 data, leave it blank, but highlight that the confirmation page must be printed and brought to the consular appointment.


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
                You are tasked with generating a Exhibit List for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page

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
            model_settings=ModelSettings(temperature=0.7),
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    max_num_results=50,
                    vector_store_ids=[vector_store_id],
                ),
            ],
        ),
        "Interview Cheat-Sheet": Agent(
            name="Interview Cheat-Sheet Agent",
            instructions=(
                f"""
                Today’s date is {current_date}.
                You are tasked with generating a  Interview Cheat-Sheet for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page


                **Step 2**: Use the following structure for the letter:
                ```
                ## Interview Cheat‑Sheet

                ### Applicant Identity & Contact Details  
                **Full Name:** [Applicant’s Full Name]  
                **Date of Birth:** [MM/DD/YYYY]  
                **Passport Number:** [Passport Number]  

                The consular officer will first verify your identity and contact information exactly as they appear on your DS‑160 form. Be prepared to state your full name, date of birth, and passport number clearly. Speak confidently and ensure you match the exact spelling and format on official documents. If your DS‑160 included any alias or previous name variations, be ready to explain briefly why you used them and confirm current legal name usage.

                ### Purpose of Trip & Planned Travel Dates  
                **Visa Category:** [e.g., B‑2 Tourist, F‑1 Student]  
                **Purpose of Visit:** [Purpose (from DS‑160)]  
                **Planned Dates of Travel:** [Arrival date] to [Departure date]  

                This section captures your stated reason for visiting the U.S. and the timeframe you expect to stay. Explain why your trip is necessary and appropriate under the specific visa category—noting, for example, tourism, business meetings, academic study, or exchange programs. If dates are approximate, clarify the rationale, such as academic term start or business schedule. Be prepared to justify why you chose that period and how it aligns with your purpose without intention of immigrant intent.

                ### U.S. Contact & Accommodation  
                **U.S. Point of Contact or Hosting Address:** [Name / Organization / Address]  
                **Accommodation Plan:** [Where you will stay]  

                If your DS‑160 provided a U.S. point of contact or address where you will reside, summarize that information. Explain the nature of your relationship with the point of contact or how you arranged your accommodation (hotel, family, host organization). Be ready to articulate why staying there is appropriate for the trip and how it supports the legitimacy of your travel purpose. Avoid speculation—state only what’s filled in your application and ensure consistency.

                ### Employment or Study Details  
                **Current Occupation or Institution:** [Employer or School Name]  
                **Job Title or Course of Study:** [Your position or program]  
                **Duration in Role or Program:** [Start Date – Present]  

                When asked, describe your current professional or academic status as provided in your DS‑160 form. Clarify your role, duties, or field of study and how long you’ve held that role or been enrolled. If applying under a student or exchange category, be sure to reference your SEVIS‑ID and educational institution details. For work or business‑based applications, explain how your trip’s activities relate to your current position or employer responsibilities.

                ### Travel History & Prior U.S. Visits  
                **Previous U.S. Visits (last 5 years):** [List dates & purpose]  
                **International Travel History:** [Other countries visited]  

                You will be asked about your recent travel history—especially to the U.S.—as entered in the DS‑160. Provide details of any past visits, including dates and legitimate reasons (tourism, work, study). If your DS‑160 recorded other international travel, be prepared to describe those trips and demonstrate credible reasons for repeated travel abroad. This shows consistency and helps eliminate concerns of immigration intent. Only speak to what’s filled in your application.

                ### Social Media & Online Presence  
                **Social Media Accounts (active within last 5 years):** [Handles or Links]  

                Modern DS‑160 versions require listing social media usernames active over the past five years. Expect to discuss these accounts if listed—why you used them and how you maintain transparency. Be ready to confirm that you included all accounts and that your online presence corresponds with the identity and background given in the DS‑160. Honesty is critical, and the consular official may check consistency between accounts and your stated personal history.

                ### Funding & Financial Ties  
                **Source of Funds for Trip:** [Personal savings / Sponsor]  
                **Employment Income or Funding Source:** [Details]  
                **Ties to Home Country:** [Assets, family, employment]  

                You may be prompted to explain how you will finance the trip and why you will return. Describe your financial means—such as salary, savings, institutional sponsorship, or family support—as indicated in your DS‑160 responses. Also emphasize ties that strongly root you in your home country, such as stable employment, family obligations, or property ownership. This underscores your intent to depart the U.S. at the end of the authorized stay.

                ### Supporting Document Prep  
                **Documents to Present:** DS‑160 confirmation page, passport, photo, receipt, supporting evidence as indicated.  

                Be ready to show your DS‑160 confirmation barcode page at the interview, along with passport, the photo upload and visa fee receipt. Gather any additional documents you listed in your DS‑160, such as itinerary, employment letters, invitation letters, educational certificates, or any other support. Organize them logically so you can refer quickly if asked. Your preparation will reflect attention to detail and reinforce credibility with the officer.

              
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
                You are tasked with generating a RFE Response for DS-160 (Nonimmigrant Visas) application.

                **Step 1**: Extract all necessary information only from the file provided. Do not use information from previous files or external sources. If any required information is missing, leave the corresponding placeholder blank; do not attempt to fill it with assumptions or unrelated data. This includes:
                - Personal details of the beneficiary or the client.
                - Employer details.
                - Job description and duties.
                - Required forms:
                     - Form DS-160, Online Nonimmigrant Visa Application
                - Supporting documents:
                    - Passport (valid for at least six months beyond the intended stay)
                    - Photo meeting DS-160 requirements
                    - Travel Itinerary
                    - Resume/CV
                    - DS-160 Confirmation Page


                **Step 2**: Use the following structure for the letter:
                ```
                # Response to Consular RFE – DS-160 Application for [Applicant’s Full Name]  
                **[Applicant’s Name or Representative's Name]**  
                **[Applicant’s Address or Attorney’s Office]**  
                **[City, State, ZIP Code]**

                **Date:** _YYYY-MM-DD_

                **Subject:** Response to Request for Evidence (RFE) – DS-160 Nonimmigrant Visa Application for [Applicant’s Full Name]

                **Dear Consular Officer,**

                **Introduction & RFE Reference**  
                **Parties & Purpose:** “[Applicant’s Name] (the “Applicant”) respectfully submits this Response to the Request for Evidence (RFE) issued in connection with the DS-160 Nonimmigrant Visa application.”  
                **RFE Details:**  
                Case Number: _[Number]_  
                RFE Issued: _[Date]_  
                Response Deadline: _[Date]_  
                **Summary of Consular Concerns:**  
                1. Discrepancy or lack of clarity in answers provided on DS-160 form.  
                2. Incomplete supporting documentation regarding identity and travel plans.  
                3. Request for further clarification on applicant’s intent and ties to home country.

                **Rebuttal to Consular Concerns**  
                **Concern 1: DS-160 Content Accuracy**  
                **Consular Position:** “Inconsistencies or vague responses in DS-160 responses.”  
                **Rebuttal:**  
                The updated DS-160 form (Ex. A) corrects and clarifies previous responses. A summary sheet outlining key changes and reasons for edits has also been included (Ex. B). The applicant affirms under oath that all details now accurately reflect their circumstances and travel intent.  

                **Concern 2: Supporting Documentation Deficiency**  
                **Consular Position:** “Insufficient documentation provided to verify identity or purpose of travel.”  
                **Rebuttal:**  
                Applicant has now submitted all required materials, including a valid passport (Ex. C), compliant photo (Ex. D), travel itinerary (Ex. E), and DS-160 confirmation page (Ex. F). Where applicable, the resume/CV (Ex. G) and any relevant employment or invitation letters have also been enclosed.  

                **Concern 3: Nonimmigrant Intent Clarification**  
                **Consular Position:** “Need for further evidence of applicant’s intent to return to home country.”  
                **Rebuttal:**  
                Applicant has provided evidence of strong ties to the home country, including family connections, property records, or ongoing educational/professional commitments (Ex. H). These demonstrate a clear intent to return after the temporary visit as stated in the application.

                **Additional Legal Authority:**  
                In accordance with the U.S. Department of State’s regulations regarding nonimmigrant visa issuance, the applicant respectfully affirms full compliance with all DS-160 requirements and guidelines as outlined at Travel.gov.

                **Conclusion & Request**  
                **Eligibility Reaffirmed:** “Based on the revised application materials and supporting documents, the Applicant has fully addressed the consular concerns and remains eligible for issuance of a nonimmigrant visa.”  
                **Request for Adjudication:** “Applicant respectfully requests that the consular officer proceed with adjudication and notify the Applicant at **[Email Address]** of the visa decision or next steps.”  
                **Point of Contact:** “For further questions or submission of additional documents, please contact **[Attorney or Applicant’s Representative Name]**, **[Title, if applicable]**, at **[Phone Number]** or **[Email Address]**.”

                **Very truly yours,**  
                \_\_\_\_\_\_\_\_\_\_\_,  
                **[Representative’s Name], [Title if applicable]**  
                **[Organization Name or Applicant]**
 

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
