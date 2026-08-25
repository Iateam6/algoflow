import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from case_jobs.integrations.openai_client import get_openai_client


logger = logging.getLogger(__name__)

GENERATION_MODEL = "gpt-5.6"


@dataclass(frozen=True)
class DocumentPrompt:
    name: str
    template: str


def format_current_date() -> str:
    return datetime.now().strftime("%B %d, %Y").replace(" 0", " ")


def build_prompt_registry():
    """Build the prompt registry for each EB-1A output document."""
    # Get today’s date in the desired format
    current_date = format_current_date()

    return {
        "I-140 Cover Letter": DocumentPrompt(
            name="I-140 Cover Letter Agent",
            template=(
                rf"""
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
                **[Date]**
                                
                **U.S. Port of Entry / USCIS Address**
                **NAFTA Free Trade Examiner**

                ### **RE: EB-1A I-140 & I-485 with Sponsors Application**

                **Employer:** [Company Name]<br>
                **Beneficiary:** [Beneficiary Full Name]<br>
                **Position:** [Job Title]<br>
                **Country:** [Country name of the beneficiary]


                Dear Free Trade Examiner:

                This letter is being submitted in support of **[Beneficiary Full Name]**’s EB-1A I-140 & I-485 with Sponsors application. **[He/She/They]** is a **[Canadian/Mexican]** national seeking EB-1A status as a **[EB-1A Profession]** under the provisions of the USMCA.

                **[Beneficiary Full Name]** qualifies for EB-1A classification by virtue of **[his/her/their]** **[Degree]** in **[Field of Study]** and substantial experience in the field. **[He/She/They]** will be employed by **[Company Name]** in the position of **[Job Title]**, where **[he/she/they]** will **[brief description of duties — 2-3 sentences]**.

                **[Beneficiary Full Name]** will receive an annual compensation of **$[Annual Salary]**. The employment is expected to begin on **[Start Date]** and continue for up to three years. **[He/She/They]** has declared the intent to depart the United States upon completion of the authorized period of stay.

                **Enclosed, please find the following materials in support of this application:**

                [Add the provided exhibit here with a short description like bellow
                EXAMPLE;
                Exhibit 1:	Form G-1450 in the amount of $2,965 for the I-907 Premium Processing fee
                            Form G-1450 in the amount of $510 for the I-129 Non-Immigrant Petition
                            Form G-1450 in the amount of $300 for the Asylum fee.
                            Form G-28 “Notice of Entry of Appearance as Attorney”
                            Form I-907 “Request for Premium Processing”
                            Form I-129 + TN Supplement “Non-Immigrant Petition for Alien Workers”
                Exhibit 2:	Copy of Applicant’s Canadian Passport as proof of his Canadian citizenship, proof of lawful stay in the U.S., and proof of ties in home country.
                Exhibit 3:   Letter of support and explanation for EB-1A qualification
                Exhibit 4:   Copy of Applicant’s educational evaluation, diploma, and transcripts
                Exhibit 5:   Copy of Applicant’s resume showing his work experience
                Exhibit 6:   Job Offer Letter with Job Description
                Exhibit 7:   Information about the Employer
                NOTE: Do not use this in the generate file covers USE the provided one
                ]

                Thank you for your assistance. Should you require any additional information, please do not hesitate to contact our office.

                **Very truly yours,**

                **[Preparer/Lawyer's Full Name], Esq.**  
                **[Firm Name]** 
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.In the "Exhibits", list only the exhibits from supporting documents are actually provided in the input. Do not list exhibits that are missing or not provided. Do not include any placeholders or blank entries for missing exhibits and the numbering of exhibits should start from 1 going on. 
                Step 6.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 7.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 8.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 9.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),            
        ),
        "Intent to Depart": DocumentPrompt(
            name="Intent to Depart",
            template=(
                rf"""
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
                **Petitioner:** [Company Name] 
                **Applicant:** [Mr./Ms.] [Beneficiary Full Name]< 
                **Nationality:** [Nationality] 
                **Classification:** EB-1A I-140 & I-485 with Sponsors

                **USCIS Filing Address / Port of Entry**

                **EXPRESSION OF INTENT TO RETURN TO MY HOME COUNTRY, [COUNTRY NAME], UPON EXPIRATION OF MY TN NON-IMMIGRANT STATUS**

                An applicant who is the beneficiary of a non-immigrant visa petition will need to satisfy the immigration officer that his or her intent is to depart the United States at the end of his or her authorized stay in TN status and not stay in the United States to adjust status or otherwise remain in the United States.

                Below is [Mr./Ms.] **[Beneficiary Full Name]**’s declaration of intent to depart the United States upon completion of his/her authorized stay in TN status.

                **I, [Beneficiary Full Name], do hereby certify, swear, or affirm under penalty of perjury as to the truth of the following statements:**

                1.I am a National and Citizen of **[Country]**. I permanently reside in **[Country of Residence]** at the following address: **[Beneficiary’s Foreign Address]**.

                2.I am applying for TN classification as a **[Job Title]** in **[Company Name]**, a U.S. Treaty Enterprise.

                3.If the United States Citizenship and Immigration Services approves my application, I understand that my stay in the United States is temporary. Thus, I intend to remain in the U.S. strictly through the duration of my authorized stay pursuant to the TN status or any extension thereof. At the end of my authorized stay in TN status, I intend to leave the United States and return to **[Country]**. I do not intend to stay in the United States to adjust status or otherwise remain in the United States regardless of legality of status.

                **Signature:**  
                \_\_\_\_\_\_\_\_\_\_\_,

                **By:** [Beneficiary Full Name]
                **Title:** [Job Title]
                **Company:** [Company Name]
                **Date:** [Date]
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 6.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 7.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 8.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),            
        ),
        "Support Letter": DocumentPrompt(
            name="Support Letter Agent",
            template=(
                rf"""

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
                **[Date]**

                **United States Customs and Border Protection / USCIS**
                
                ### **RE: Petition for EB-1A I-140 & I-485 with Sponsors Status for Mr. / Ms. [Beneficiary Full Name]**
                
                **Profession (USMCA, formerly NAFTA):** [Job Title]
                
                Dear Sir or Madam:
                
                This letter is written in support of Mr. / Ms. **[Beneficiary Full Name]**'s petition for EB-1A I-140 & I-485 with Sponsors status pursuant to the USMCA, formerly North American Free Trade Agreement ("NAFTA"). **[Company Name]** has made an offer of temporary employment to Mr. / Ms. **[Beneficiary Full Name]** in the position of **[Job Title]** (under the USMCA classification of **[Job Title]**) with a monthly base salary of $[Monthly Salary] which translates to **$[Annual Salary]** annually. We are seeking the approval of his/her petition to enable **[Company Name]** to employ him/her immediately upon the approval of his/her EB-1A I-140 & I-485 with Sponsors visa, up to three years.
                
                ### BACKGROUND OF THE EMPLOYER
                
                **[Company Name]**
                
                **[Company Name]** is **[full company description]**. (See the exhibit(s) containing the employer background information)
                
                ### Job Title
                
                **[Company Name]** has offered Mr. / Ms. **[Beneficiary Full Name]** the position of **[Job Title]** in which capacity he/she will serve at the pleasure and assignment of **[Name of Immediate Supervisor]**. In his/her position he/she will be responsible for the following duties:
                
                **[Duty 1 in bulet point]**
                **[Duty 2 in bulet point]**
                **[Duty 3 in bulet point]**
                **[Duty 4 in bulet point]**
                **[Duty 5 in bulet point]**
                **[Duty 6 in bulet point]**
                **[Duty 7 in bulet point]**
                **[Duty 8 in bulet point]**
                **[Duty 9 in bulet point]**
                **[Duty 10 in bulet point]**
                
                This is a strategic and consultative role designed to support the growth and cohesion of **[Company Name]**'s efforts. (See the exhibit(s) containing the job offer letter, job duties and responsibilities, and consulting agreement)
                
                ### APPLICANT'S QUALIFICATIONS
                
                Mr. / Ms. **[Beneficiary Full Name]** holds a bachelor's degree in **[Degree]** with a major in **[Major]** from **[University Name]**, providing a strong academic foundation. **[His/Her]** career spans diverse leadership roles. **[Brief professional summary]**.
                
                Mr. / Ms. **[Beneficiary Full Name]**'s extensive background in **[key areas]** positions him/her as a highly capable candidate for a **[Job Title]** role. (See the exhibit(s) containing Mr. / Ms. **[Beneficiary Full Name]**'s educational and professional credentials)
                
                ### CONCLUSION
                
                It is evident that Mr. / Ms. **[Beneficiary Full Name]** is eligible for the EB-1A visa as a **[Job Title]** as he/she possesses the appropriate degree and a considerable number of years of experience. **[Company Name]** offers Mr. / Ms. **[Beneficiary Full Name]** employment immediately upon the approval of his/her EB-1A visa, up to three years. He/She will be compensated at a monthly base salary of $[Monthly Salary] which translates to **$[Annual Salary]** annually.
                
                For the foregoing reasons, we respectfully request your favorable adjudication of his/her EB-1A petition on behalf of Mr. / Ms. **[Beneficiary Full Name]**.
                
                **Very truly yours,**
                
                **[Name of Immediate Supervisor]**  
                **[Position]**  
                **[Company Name]**  
                **[Contact Information]**
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 6.Ensure the tone is professional and concise. Enclose the entire letter within triple backticks like this: ``` Your letter content here ```.
                Step 7.Each and every point should be elaborated in detail in about 100 words and don't leave section of the letter out it it a legal file.
                Step 8.Leave the back‐slashed underscores exactly as written—do not remove the backslashes.
                """
            ),           
        ),
        "Recommendation-Letter": DocumentPrompt(
            name="Recommendation Letter Agent",
            template=(
                rf"""
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
        ),
        "Exhibit List": DocumentPrompt(
            name="Exhibit List",
            template=(
                rf"""

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
                **List of Supporting Documents**  
                **EB-1A I-140 & I-485 with Sponsors Visa Application**

                **Petitioner:** [COMPANY NAME]  
                **Beneficiary:** [BENEFICIARY’S NAME]

                Pursuant to the United States-Mexico-Canada Agreement, a foreign national is entitled to enter the United States under the EB-1A I-140 & I-485 with Sponsors Status Visa category. Below is a complete list of supporting documents submitted to establish that **Mr. / Ms. [BENEFICIARY’S NAME]**, a citizen of **[Canada / Mexico]**, is qualified for the EB-1A Visa.

                | **Exhibit 1** | **Forms & Fees** |
                |---------------|------------------|
                |               | 1.1 Cover Letter |
                |               | 1.2 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.3 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.4 Form G-1450/G-1650 in the amount of **$[Amount]** |
                |               | 1.5 Form G-28 “Notice of Entry of Appearance as Attorney” |
                |               | 1.6 Form I-907 “Request for Premium Processing” |
                |               | 1.7 Form I-129 + EB-1A I-140 & I-485 with Sponsors Supplement “Petition for a Nonimmigrant Worker” |
                | **Exhibit 2** | **Proof of Nationality & Lawful Stay in the U.S. and Dependents** |
                |---------------|------------------------------------------------------------------|
                |               | 2.1 [BENEFICIARY’S NAME] – Passport |
                |               | 2.2 [BENEFICIARY’S NAME] – Previous Visa |
                |               | 2.3 [BENEFICIARY’S NAME] – Birth Certificate |
                |               | 2.4 [BENEFICIARY’S NAME] – Resume |
                |               | 2.5 [BENEFICIARY’S NAME] – Educational Credentials |
                |               | 2.6 [DEPENDENT SPOUSE NAME] – Passport |
                |               | 2.7 [DEPENDENT SPOUSE NAME] – Marriage Certificate |
                |               | 2.8 [DEPENDENT SPOUSE NAME] – Birth Certificate |
                |               | 2.9 [CHILD NAME] – Passport |
                |               | 2.10 [CHILD NAME] – Birth Certificate |
                | **Exhibit 3** | **Support Letter** |
                |---------------|--------------------|
                |               | 3.1 [COMPANY NAME] – Letter in Support of the Beneficiary |
                | **Exhibit 4** | **Applicant’s Educational Credentials** |
                |---------------|-----------------------------------------|
                |               | 4.1 Beneficiary’s Diploma |
                |               | 4.2 Beneficiary’s Certifications |
                |               | 4.3 Beneficiary’s Transcript of Records |
                | **Exhibit 5** | **Applicant’s Work Experience** |
                |---------------|---------------------------------|
                |               | 5.1 Curriculum Vitae / Resume |
                |               | 5.2 Certification of Employment |
                |               | 5.3 Trainings |
                | **Exhibit 6** | **Offer Letter with Summary of Terms** |
                |---------------|----------------------------------------|
                |               | 6.1 [COMPANY NAME] – Petitioner’s Job Offer |
                | **Exhibit 7** | **Employer’s Background Information** |
                |---------------|---------------------------------------|
                |               | 7.1 Articles of Incorporation / Organization |
                |               | 7.2 IRS – EIN Confirmation Number |
                |               | 7.3 Company Activities |
                |               | 7.4 Office Photos / Company Website |
                | **Exhibit 8** | **Intent to Depart** |
                |---------------|----------------------|
                |               | 8.1 Beneficiary’s Declaration of Intent to Depart |
                ```
                step 3.While selecting data to fill in the placeholders, use only accurate and relevant information from the provided input file or files. If the required information is not available, leave the placeholder blank. Do not attempt to fill placeholders with incorrect or unrelated data.
                Step 4.Adopt a professional, concise, firm tone—polite but unequivocal—avoiding needless legalese.
                Step 5.Output raw Markdown only: use headings (`#`, `##`, `###`), bold for labels, lists for items, and blank lines for paragraphs. Do not wrap in backticks or code fences—just feed it straight to Pandoc.
                Step 6.Ensure the tone is professional and concise. Enclose the entire exhibit is within triple backticks like this: ``` Your Exhibit content here ```.               
                """
            ),
        ),
        "Evidence-Summary Chart": DocumentPrompt(
            name="Evidence-Summary Chart Agent",
            template=(
                rf"""
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
        ),
        "RFE Response Brief": DocumentPrompt(
            name="RFE Response Brief",
            template=(
                rf"""
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
        ),
        "Demand Letter": DocumentPrompt(
            name="Demand Letter",
            template=(
                        rf"""
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
        ),
        "Assessment Report": DocumentPrompt(
            name="Assessment Report",
            template=(
                rf"""

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
        ),
        "Visa Application Summary Report": DocumentPrompt(
            name="Visa Application Summary Report",
            template=(
                rf"""
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
        ),
    }

RETRIEVAL_HINTS = {
    "I-140 Cover Letter": [
        # Header block
        "beneficiary full name",
        "sponsor name employer petitioner",
        "EB-1A extraordinary ability",
        "I-140 petition I-485 adjustment of status",
        "beneficiary country of citizenship nationality",
        
        # Beneficiary identity & pronoun
        "beneficiary gender pronoun he she they",
        "citizenship nationality passport",
        
        # Sponsor information
        "sponsor company name employer",
        "sponsor relationship to beneficiary",
        "job offer from sponsor",
        "sponsor support letter",
        
        # Field of expertise & classification
        "field of expertise area of extraordinary ability",
        "sustained national or international acclaim",
        "one of the small percentage at the very top of the field",
        
        # Regulatory criteria
        "awards prizes recognition",
        "membership associations outstanding achievements",
        "published material major media",
        "original contributions major significance",
        "scholarly articles publications citations",
        "leading or critical role distinguished organizations",
        "high salary remuneration",
        "commercial success",
        "judging the work of others",
        
        # I-485 related
        "adjustment of status I-485",
        "priority date",
        "supporting documentation for adjustment",
        
        # Exhibits
        "submitted exhibits list supporting documents",
        
        # Signature block
        "attorney preparer name law firm name",
    ],
    "Intent to Depart": [
        # Header block
        "beneficiary full name",
        "sponsor name employer",
        "EB-1A extraordinary ability",
        "I-140 I-485",
        "beneficiary country of citizenship nationality",
        
        # Identity & country
        "citizenship nationality country",
        "passport biographical pages passport copy",
        
        # Field & future work
        "field of expertise area of extraordinary ability",
        "intend to continue work in the area of extraordinary ability",
        "prospective work United States",
        "work will benefit the United States",
        
        # Sponsor-related
        "job offer from sponsor",
        "position offered by sponsor",
        "sponsor support",
        
        # Credentials
        "degree certificate diploma transcript resume",
        "beneficiary qualifications professional background",
        "publications citations awards patents",
        "letters of recommendation expert opinion",
        
        # Residence / ties
        "foreign home address permanent residence",
        "family ties property ownership",
    ],
    "Support Letter": [
        # Document filenames
        "sponsor support letter",
        "expert recommendation letters",
        "letters of recommendation expert opinion",
        "resume curriculum vitae CV",
        "degree certificate diploma transcript",
        
        # Sponsor & beneficiary
        "sponsor company name employer",
        "beneficiary full name",
        "relationship between sponsor and beneficiary",
        "job title position offered by sponsor",
        
        # Field of expertise
        "field of expertise area of extraordinary ability",
        "sustained national or international acclaim",
        
        # Criteria evidence
        "awards prizes recognition",
        "membership associations",
        "published material major media",
        "original contributions major significance",
        "scholarly articles citations",
        "leading or critical role",
        "high salary remuneration",
        "commercial success",
        "judging the work of others",
        
        # Professional background
        "professional experience career summary",
        "key areas of expertise",
        "publications citations patents awards",
        
        # Education
        "degree major field of study university",
        "transcript educational credentials",
        
        # Future work
        "continued work in the field",
        "work will benefit the United States",
    ],
    "Recommendation-Letter": [
        "recommendation letter",
        "expert letter",
        "EB-1A",
        "extraordinary ability",
        "original contributions",
        "major significance",
        "critical role",
        "awards-recognition",
        "media-reports",
        "recommendation-letters",
        "degree-evidence",
    ],
    "Exhibit List": [
        "exhibit list",
        "EB-1A",
        "i-140",
        "g-1145",
        "g-28-company",
        "i-907",
        "all-degree-certs",
        "awards-recognition",
        "degree-evidence",
        "form-i-94",
        "passport",
        "visa-pages",
        "recommendation-letters",
        "media-reports",
        "social-security-card",
        "permanent-resident-card",
        "petitioner-pay-stubs",
    ],
    "Evidence-Summary Chart": [
        "evidence summary chart",
        "EB-1A criteria",
        "extraordinary ability",
        "awards-recognition",
        "media-reports",
        "recommendation-letters",
        "all-degree-certs",
        "degree-evidence",
        "petitioner-pay-stubs",
        "permanent-resident-card",
        "social-security-card",
    ],
    "RFE Response Brief": [
        "request for evidence response",
        "RFE",
        "EB-1A",
        "I-140",
        "USCIS",
        "extraordinary ability",
        "sustained acclaim",
        "major significance",
        "awards-recognition",
        "media-reports",
        "recommendation-letters",
        "degree-evidence",
    ],
    "Demand Letter": [
        "demand letter",
        "pending adjudication",
        "adjudication delay",
        "EB-1A",
        "I-140",
        "USCIS",
        "i-907",
        "g-28-company",
        "filing receipt",
        "final decision",
    ],
    "Assessment Report": [
        "assessment report",
        "EB-1A",
        "I-140",
        "extraordinary ability",
        "sustained acclaim",
        "EB-1A criteria",
        "all-degree-certs",
        "awards-recognition",
        "degree-evidence",
        "media-reports",
        "recommendation-letters",
        "petitioner-pay-stubs",
    ],
    "Visa Application Summary Report": [
        "visa application summary report",
        "EB-1A",
        "I-140",
        "i-140",
        "g-1145",
        "g-28-company",
        "i-907",
        "form-i-94",
        "passport",
        "visa-pages",
        "all-degree-certs",
        "recommendation-letters",
        "media-reports",
        "social-security-card",
        "permanent-resident-card",
    ],
}


def build_retrieval_query(file_type: str) -> str:
    hints = RETRIEVAL_HINTS.get(file_type, [])
    return " | ".join([file_type, *hints])


def deduplicate_retrieved_context(retrieved_context) -> list:
    deduplicated = []
    seen_keys = set()

    for document in retrieved_context:
        metadata = getattr(document, "metadata", {}) or {}
        key = (
            str(metadata.get("file_hash", "")),
            str(metadata.get("page_number", "")),
            str(metadata.get("chunk_index", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append(document)

    return deduplicated


def build_retrieved_case_record(retrieved_context) -> str:
    sections = []
    for index, document in enumerate(deduplicate_retrieved_context(retrieved_context), start=1):
        metadata = getattr(document, "metadata", {}) or {}
        sections.append(
            "\n".join(
                [
                    f"### Retrieved Chunk {index}",
                    f"- Source: {metadata.get('source_name', 'unknown')}",
                    f"- Category: {metadata.get('source_category', 'unknown')}",
                    f"- Page: {metadata.get('page_number', 'unknown')}",
                    f"- Chunk: {metadata.get('chunk_index', 'unknown')}",
                    f"- Extraction mode: {metadata.get('extraction_mode', 'unknown')}",
                    document.page_content.strip(),
                ]
            ).strip()
        )

    if not sections:
        return "No retrieved case record was available."

    return "\n\n".join(sections)


def summarise_source_manifest(source_manifest: list[dict]) -> str:
    lines = []
    for entry in source_manifest:
        lines.append(
            "\n".join(
                [
                    f"- Source name: {entry.get('original_filename', 'unknown')}",
                    f"  Category: {entry.get('name', 'unknown')}",
                    f"  File hash: {entry.get('file_hash', 'unknown')}",
                    f"  Extension: {entry.get('extension', 'unknown')}",
                    f"  MIME type: {entry.get('content_type', 'unknown')}",
                    f"  Extraction mode: {entry.get('extraction_mode', 'unknown')}",
                    f"  Pages: {entry.get('page_count', 'unknown')}",
                ]
            )
        )

    return "\n".join(lines) if lines else "- No source manifest available."


def get_document_template(file_type: str) -> str:
    prompt = build_prompt_registry().get(file_type)
    if not prompt:
        raise ValueError(f"No prompt found for document type: {file_type}")
    return prompt.template.strip()


def build_generation_prompt(file_type: str, retrieved_context, source_manifest: list[dict]) -> str:
    prompt_registry = build_prompt_registry()
    prompt = prompt_registry.get(file_type)
    if not prompt:
        raise ValueError(f"No prompt found for document type: {file_type}")

    return "\n\n".join(
        [ 
            prompt.template.strip(),
            "# Retrieved Case Record",
            build_retrieved_case_record(retrieved_context),
            "# Source Manifest",
            summarise_source_manifest(source_manifest),
            "# Additional Output Rules",
            "Treat the template as a structural guide only; do not copy it verbatim.",
            "Act like a lawyer: analyze the retrieved case record and the source manifest, then draft a document grounded in those materials.",
            "Use the retrieved case record as the primary source of factual support and the source manifest as supporting evidence.",
            "When a template field or placeholder is not directly available, look for equivalent or related evidence in the retrieved case record/source manifest and use that to fill the section.",
            "For example, if the template asks for 'petitioner' or 'employer' and the case record uses a different but equivalent term, use the correct party from the evidence.",
            "Fill in every relevant section with facts supported by the retrieved case record or source manifest; if evidence is missing, leave the relevant content blank or mark it as [Not provided] rather than inventing facts.",
            "If key facts are missing, leave the relevant placeholders blank.",
            "Return only the final document enclosed in triple backticks.",
        ]
    ).strip()


async def generate_document(file_type, retrieved_context, source_manifest):
    prompt_text = build_generation_prompt(file_type, retrieved_context, source_manifest)
    client = get_openai_client()
    logger.info("Generating %s with %s retrieved chunks", file_type, len(retrieved_context))

    response = await asyncio.to_thread(
        client.responses.create,
        model=GENERATION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    }
                ],
            }
        ],
    )
    return (response.output_text or "").strip()

