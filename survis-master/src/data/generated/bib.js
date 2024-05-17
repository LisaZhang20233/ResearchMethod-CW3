const generatedBibEntries = {
    "A Deep Dive into Large Language Models for Automated Bug Localization and Repair": {
        "abstract": "Large language models (LLMs) have shown impressive effectiveness in various software engineering tasks, including automated program repair (APR). In this study, we take a deep dive into automated bug fixing utilizing LLMs. In contrast to many deep learning-based APR methods that assume known bug locations, rely on line-level localization tools, or address bug prediction and fixing in one step, our approach uniquely employs LLMs to predict bug location at the token level and subsequently utilizes them for bug fixing. This methodological separation of bug localization and fixing using different LLMs enables effective integration of diverse contextual information and improved incorporation of inductive biases. We introduce Toggle: Token-Granulated Bug Localization and Repair, a comprehensive program repair framework that integrates a bug localization model, an adjustment unit, and a bug-fixing model. Toggle takes a buggy function as input and generates a complete corrected function. We investigate various styles of prompting to the bug fixing model to identify the most effective prompts that better utilize the inductive bias and significantly outperform others. Toggle achieves the new state-of-the-art (SOTA) performance on the CodeXGLUE code refinement benchmark, and exhibits better and comparable performance on several other widely-used APR datasets, including Defects4J.",
        "author": "Soneya Binta Hossain, Nan Jiang, Qiang Zhou, Xiaopeng Li, Wen-Hao Chiang, Yingjun Lyu, Hoan Nguyen, Omer Tripp",
        "doi": "10.48550/arXiv.2404.11595",
        "keywords": "Automated Bug Localization and Fix, Large Language Models",
        "title": " A Deep Dive into Large Language Models for Automated Bug Localization and Repair",
        "type": "article",
        "url": "https://arxiv.org/pdf/2404.11595",
        "year": "2024"
    },
    "AGENTFL: Scaling LLM-based Fault Localization to Project-Level Context": {
        "abstract": "Software development is a complex activity requiring intelligent action. This paper explores the use of an AI technique for one step in software development, viz. detecting the location of a fault in a program. A measure of program progress is proposed, which uses a Na\u00efve Bayes model to measure how useful the information that has been produced by the program to the task that the program is tackling. Then, deviations in that measure are used to find the location of faults in the code. Experiments are carried out to test the effectiveness of this measure.",
        "author": "Yihao Qin, Shangwen Wang, Yiling Lou, Jinhao Dong, Kaixin Wang, Xiaoling Li, Xiaoguang Mao",
        "doi": "10.48550/arXiv.2403.16362",
        "keywords": "Large Language Model, Fault Localization",
        "title": " AGENTFL: Scaling LLM-based Fault Localizationto Project-Level Context",
        "type": "article",
        "url": "https://arxiv.org/pdf/2403.16362",
        "year": "2024"
    },
    "Bug Localization with Combination of Deep Learning and Information Retrieval": {
        "abstract": "The automated task of locating the potential buggy files in a software project given a bug report is called bug localization. Bug localization helps developers focus on crucial files. However, the existing automated bug localization approaches face a key challenge, called lexical mismatch. Specifically, the terms used in bug reports to describe a bug are different from the terms and code tokens used in source files. To address that, we present a novel approach that uses deep neural network (DNN) in combination with rVSM, an information retrieval (IR) technique. rVSM collects the feature on the textual similarity between bug reports and source files. DNN is used to learn to relate the terms in bug reports to potentially different code tokens and terms in source files. Our empirical evaluation on real-world bug reports in the open-source projects shows that DNN and IR complement well to each other to achieve higher bug localization accuracy than individual models. Importantly, our new model, DNNLOC, with a combination of the features built from DNN, rVSM, and project's bug-fixing history, achieves higher accuracy than the state-of-the-art IR and machine learning techniques. In half of the cases, it iscorrect with just a single suggested file. In 66% of the time, acorrect buggy file is in the list of three suggested files. With 5 suggested files, it is correct in almost 70% of the cases.",
        "author": "A. N. Lam, A. T. Nguyen, H. A. Nguyen and T. N. Nguyen",
        "doi": "10.1109/ICPC.2017.24",
        "journal": "2017 IEEE/ACM 25th International Conference on Program Comprehension (ICPC)",
        "keywords": "Computer bugs;Feature extraction;Metadata;Neural networks;Computer architecture;Training;Software;Bug Localization;Deep Learning;Code Retrieval;Information Retrieval",
        "publisher": "Buenos Aires, Argentina,",
        "title": " Bug Localization with Combination of Deep Learning and Information Retrieval",
        "type": "article",
        "url": "https://ieeexplore.ieee.org/abstract/document/7961519",
        "year": "2017"
    },
    "DeepBugs: a learning approach to name-based bug detection": {
        "abstract": "Natural language elements in source code, e.g., the names of variables and functions, convey useful information.However, most existing bug detection tools ignore this information and therefore miss some classes of bugs.The few existing name-based bug detection approaches reason about names on a syntactic level and rely on manually designed and tuned algorithms to detect bugs. This paper presents DeepBugs, a learning approach to name-based bug detection, which reasons about names based on a semantic representation and which automatically learns bug detectors instead of manually writing them. We formulate bug detection as a binary classification problem and train a classifier that distinguishes correct from incorrect code. To address the challenge that effectively learning a bug detector requires examples of both correct and incorrect code, we create likely incorrect code examples from an existing corpus of code through simple code transformations. A novel insight learned from our work is that learning from artificially seeded bugs yields bug detectors that are effective at finding bugs in real-world code. We implement our idea into a framework for learning-based and name-based bug detection. Three bug detectors built on top of the framework detect accidentally swapped function arguments, incorrect binary operators, and incorrect operands in binary operations. Applying the approach to a corpus of 150,000 JavaScript files yields bug detectors that have a high accuracy (between 89% and 95%), are very efficient (less than 20 milliseconds per analyzed file), and reveal 102 programming mistakes (with 68% true positive rate) in real-world code.",
        "author": "MICHAEL PRADEL, KOUSHIK SEN",
        "doi": "10.1145/3276517",
        "journal": "Proceedings of the ACM on Programming Languages",
        "keywords": "Bug detection, Natural language, Machine learning, Name-based program analysis, JavaScript",
        "number": "147",
        "title": "DeepBugs: a learning approach to name-based bug detection",
        "type": "article",
        "url": "https://v dl.acm.org/doi/pdf/10.1145/3276517",
        "volume": "2",
        "year": "2018"
    },
    "Detect-Localize-Repair: A Unified Framework for Learning to Debug with CodeT5": {
        "abstract": "Automated software debugging is a crucial task for improving the productivity of software developers. Many neural-based techniques have been proven effective for debugging-related tasks such as bug localization and program repair (or bug fixing). However, these techniques often focus only on either one of them or approach them in a stage-wise manner, ignoring the mutual benefits between them. In this work, we propose a novel unified \\emph{Detect-Localize-Repair} framework based on a pretrained programming language model CodeT5 to seamlessly address these tasks, named CodeT5-DLR. Specifically, we propose three objectives to adapt the generic CodeT5 for debugging: a bug detection objective to determine whether a given code snippet is buggy or not, a bug localization objective to identify the buggy lines, and a program repair objective to translate the buggy code to its fixed version. We evaluate it on each of these tasks and their combined setting on two newly collected line-level debugging datasets in Java and Python. Extensive results show that our model significantly outperforms existing baselines from both NLP and software engineering domains.",
        "author": "Nghi D. Q. Bui, Yue Wang, Steven Hoi",
        "doi": "10.48550/arXiv.2211.14875",
        "keywords": "Bug detecting,CodeT5",
        "title": " Detect-Localize-Repair: A Unified Framework for Learning to Debug with CodeT5",
        "type": "article",
        "url": "https://arxiv.org/abs/2211.14875",
        "year": "2022"
    },
    "Language Models are Better Bug Detector Through Code-Pair Classification": {
        "abstract": "Large language models (LLMs) such as GPT-3.5 and CodeLlama are powerful models for code generation and understanding. Fine-tuning these models comes with a high computational cost and requires a large labeled dataset. Alternatively, in-context learning techniques allow models to learn downstream tasks with only a few examples. Recently, researchers have shown how in-context learning performs well in bug detection and repair. In this paper, we propose code-pair classification task in which both the buggy and non-buggy versions are given to the model, and the model identifies the buggy ones. We evaluate our task in real-world dataset of bug detection and two most powerful LLMs. Our experiments indicate that an LLM can often pick the buggy from the non-buggy version of the code, and the code-pair classification task is much easier compared to be given a snippet and deciding if and where a bug exists.",
        "author": "Kamel Alrashedy, Ahmed Binjahlan",
        "doi": "10.48550/arXiv.2311.07957",
        "keywords": "Language model, Code-pair Classification",
        "title": " Language Models are Better Bug Detector Through Code-Pair Classification",
        "type": "article",
        "url": "https://arxiv.org/abs/2311.07957",
        "year": "2023"
    },
    "Self-Supervised Bug Detection and Repair": {
        "abstract": "Machine learning-based program analyses have recently shown the promise of integrating formal and probabilistic reasoning towards aiding software development. However, in the absence of large annotated corpora, training these analysis challenging. Towards addressing this, we present BUGLAB, an approach for self-supervised learning of bug detection and repair. BUGLAB co-trains two models: (1) a detector model that learns to detect and repair bugs in code, (2) a selector model that learns to create buggy code for the detector to use as training data. A Python implementation of BUGLAB improves by up to 30% upon baseline methods on a test dataset of 2374 real-life bugs and finds 19 previously unknown bugs in open-source software.",
        "author": "Miltiadis Allamanis, Henry Jackson-Flux, Marc Brockschmidt",
        "keywords": "Bug detecting, Machine Learning",
        "publisher": "Curran Associates, Inc.",
        "title": " Self-Supervised Bug Detection and Repair",
        "type": "article",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2021/file/ea96efc03b9a050d895110db8c4af057-Paper.pdf",
        "volume": "34",
        "year": "2021"
    },
    "Software Fault Localisation via Probabilistic Modelling": {
        "abstract": "Software development is a complex activity requiring intelligent action. This paper explores the use of an AI technique for one step in software development, viz. detecting the location of a fault in a program. A measure of program progress is proposed, which uses a Na\u00efve Bayes model to measure how useful the information that has been produced by the program to the task that the program is tackling. Then, deviations in that measure are used to find the location of faults in the code. Experiments are carried out to test the effectiveness of this measure.",
        "author": "Colin G. Johnson",
        "doi": "10.1007/978-3-030-63799-6_20",
        "journal": "Artificial Intelligence XXXVII. SGAI 2020.",
        "keywords": "Software development, bug finding, Machine Learning",
        "number": "01",
        "publisher": "Springer, Cham",
        "series": "LNAI",
        "title": " Software Fault Localisation via Probabilistic Modelling",
        "type": "article",
        "url": "https://www.colinjohnson.me.uk/research/papers/Johnson_SGAI2020.pdf",
        "volume": "12498",
        "year": "2020"
    },
    "Using Distributed Representation of Code for Bug Detection": {
        "abstract": "Recent advances in neural modeling for bug detection have been very promising. More specifically, using snippets of code to create continuous vectors or \\textit{embeddings} has been shown to be very good at method name prediction and claimed to be efficient at other tasks, such as bug detection. However, to this end, the method has not been empirically tested for the latter. In this work, we use the Code2Vec model of Alon et al. to evaluate it for detecting off-by-one errors in Java source code. We define bug detection as a binary classification problem and train our model on a large Java file corpus containing likely correct code. In order to properly classify incorrect code, the model needs to be trained on false examples as well. To achieve this, we create likely incorrect code by making simple mutations to the original corpus. Our quantitative and qualitative evaluations show that an attention-based model that uses a structural representation of code can be indeed successfully used for other tasks than method naming.",
        "author": "J\u00f3n Arnar Briem, Jordi Smit, Hendrig Sellik, Pavel Rapoport",
        "doi": "10.48550/arXiv.1911.12863",
        "keywords": "Distributed Representation, Bug Detecting",
        "title": " Using Distributed Representation of Code for Bug Detection",
        "type": "article",
        "url": "https://arxiv.org/abs/1911.12863",
        "year": "2019"
    },
    "WitheredLeaf: Finding Entity-Inconsistency Bugs with LLMs": {
        "abstract": "riginating from semantic bugs, Entity-Inconsistency Bugs (EIBs) involve misuse of syntactically valid yet incorrect program entities, such as variable identifiers and function names, which often have security implications. Unlike straightforward syntactic vulnerabilities, EIBs are subtle and can remain undetected for years. Traditional detection methods, such as static analysis and dynamic testing, often fall short due to the versatile and context-dependent nature of EIBs. However, with advancements in Large Language Models (LLMs) like GPT-4, we believe LLM-powered automatic EIB detection becomes increasingly feasible through these models' semantics understanding abilities. This research first undertakes a systematic measurement of LLMs' capabilities in detecting EIBs, revealing that GPT-4, while promising, shows limited recall and precision that hinder its practical application. The primary problem lies in the model's tendency to focus on irrelevant code snippets devoid of EIBs. To address this, we introduce a novel, cascaded EIB detection system named WitheredLeaf, which leverages smaller, code-specific language models to filter out most negative cases and mitigate the problem, thereby significantly enhancing the overall precision and recall. We evaluated WitheredLeaf on 154 Python and C GitHub repositories, each with over 1,000 stars, identifying 123 new flaws, 45% of which can be exploited to disrupt the program's normal operations. Out of 69 submitted fixes, 27 have been successfully merged.",
        "author": "Hongbo Chen, Yifan Zhang, Xing Han, Huanyao Rong, Yuheng Zhang, Tianhao Mao, Hang Zhang, XiaoFeng Wang, Luyi Xing, Xun Chen",
        "doi": "10.48550/arXiv.2405.01668",
        "title": " WitheredLeaf: Finding Entity-Inconsistency Bugs with LLMs",
        "type": "article",
        "url": "https://arxiv.org/pdf/2405.01668",
        "year": "2024"
    }
};