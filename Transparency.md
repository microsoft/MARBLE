## Transparency

### Intended uses
MARBLE is best suited for researchers in the computational pathology field who are interested in quantifying biomarkers using hematoxylin and eosin stained images.

MARBLE is being shared with the research community to facilitate reproduction of our results and foster further research in this area.

MARBLE is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

### Out-of-scope uses
MARBLE is not well suited for applications outside of computational pathology.

We do not recommend using MARBLE in commercial or real-world applications without further testing and development. It is being released for research purposes and is not for diagnostic use. Predictions should not be treated as a substitute for validated diagnostic methods.

MARBLE was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.

Without further testing and development, MARBLE should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.

We do not recommend using MARBLE in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).

### Limitations
MARBLE was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.

MARBLE has only been tested on one biomarker panel and one institution; users should evaluate MARBLE on their own data and assess its performance accordingly.

Because of lack of access to metadata, MARBLE was not evaluated for bias/fairness across demographic or other subgroups.

### Best practices
MARBLE was only tested using image representations from Virchow2. Users should proceed with caution as they update and reconsider components of the design as new components may lead to unexpected results.

We strongly encourage users to use LLMs/MLLMs that support robust Responsible AI mitigations, such as Azure Open AI (AOAI) services. Such services continually update their safety and RAI mitigations with the latest industry standards for responsible use. For more on AOAI’s best practices when employing foundations models for scripts and applications:
- [What is Azure AI Content Safety?](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)  
- [Overview of Responsible AI practices for Azure OpenAI models](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview) 
- [Azure OpenAI Transparency Note](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note)
- [OpenAI’s Usage policies](https://openai.com/policies/usage-policies) 
- [Azure OpenAI’s Code of Conduct](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct) 

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.   

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with MARBLE. 

It is the user’s responsibility to ensure that the use of MARBLE complies with relevant data protection regulations and organizational guidelines.

Developers should follow transparency best practices and inform end-users they are interacting with an AI system.
