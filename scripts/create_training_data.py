#!/usr/bin/env python3
"""
Generate diverse conversational training data for KhatriVoice.
Creates meaningful, varied Q&A pairs across multiple topics.
"""

import random

# Comprehensive Q&A pairs organized by topic
BASE_CONVERSATIONS = [
    # GREETINGS & BASIC INTERACTIONS
    ("Hi", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What can I do for you?"),
    ("Hey", "Hey! Nice to hear from you. What's on your mind?"),
    ("Good morning", "Good morning! Hope you're having a great day!"),
    ("Good afternoon", "Good afternoon! How can I assist you?"),
    ("Good evening", "Good evening! What can I help you with?"),
    ("How are you?", "I'm doing well, thanks for asking! How about you?"),
    ("What's up?", "Not much, just here to help! What do you need?"),
    ("Thanks", "You're welcome! Let me know if you need anything else."),
    ("Thank you", "Happy to help! Feel free to ask more questions."),
    ("That helps", "Great! I'm glad it was useful."),
    ("I understand now", "Excellent! Let me know if you have other questions."),
    ("Goodbye", "Goodbye! Have a wonderful day!"),
    ("Bye", "Bye! Take care!"),
    ("See you later", "See you! Come back anytime."),

    # PROGRAMMING FUNDAMENTALS
    ("What is programming?", "Programming is writing instructions that tell computers how to perform tasks. It involves creating algorithms and implementing them in languages like Python or JavaScript."),
    ("What is a variable?", "A variable is a named container that stores data in memory. You can think of it like a labeled box that holds a value."),
    ("What is a function?", "A function is a reusable block of code that performs a specific task. Functions help organize code and avoid repetition."),
    ("What is a loop?", "A loop is a construct that repeats code multiple times. Common types include for-loops and while-loops."),
    ("What is an if statement?", "An if statement executes code only when a condition is true. It allows programs to make decisions."),
    ("What is an array?", "An array stores multiple values in a single variable. Elements are accessed by their index position."),
    ("What is debugging?", "Debugging is finding and fixing errors in code. It involves identifying where problems are and correcting them."),
    ("What is a bug?", "A bug is an error or flaw that causes programs to behave unexpectedly."),
    ("What is compilation?", "Compilation converts source code into machine-readable instructions."),
    ("What is an algorithm?", "An algorithm is a step-by-step procedure for solving a problem."),
    ("What is pseudocode?", "Pseudocode is an informal way to describe an algorithm using plain language."),
    ("What are code comments?", "Comments are text notes in code that explain what the code does."),

    # PYTHON PROGRAMMING
    ("What is Python?", "Python is a high-level programming language known for its simple, readable syntax. It's used for web development, data science, and machine learning."),
    ("Why is Python popular?", "Python is popular because it's easy to learn, has a large ecosystem of libraries, and can be used for many different projects."),
    ("What is a Python list?", "A list in Python is an ordered collection that can hold items of different types using square brackets."),
    ("What is a Python dictionary?", "A dictionary is a collection of key-value pairs using curly braces."),
    ("How do I define a function in Python?", "Use the def keyword followed by the function name. For example: def my_function(): defines a function."),
    ("What is Python used for?", "Python is used for web development, data analysis, artificial intelligence, automation, and more."),
    ("What is pip in Python?", "pip is Python's package manager for installing libraries from the Python Package Index."),
    ("What is a Python module?", "A module is a file containing Python code that can be imported into other programs."),
    ("How do I read a file in Python?", "Use the open() function with a file path to read file contents."),
    ("What is list comprehension?", "List comprehension is a concise way to create lists in a single line of code."),
    ("What are Python decorators?", "Decorators are functions that modify the behavior of other functions using the @decorator syntax."),
    ("What is the difference between list and tuple?", "Lists are mutable and can be changed. Tuples are immutable and cannot be modified after creation."),

    # JAVASCRIPT & WEB
    ("What is JavaScript?", "JavaScript is a programming language that runs in web browsers, making pages interactive. It can also run on servers with Node.js."),
    ("What is the difference between JavaScript and Java?", "JavaScript and Java are completely different languages. JavaScript is for web development, Java is a general-purpose language."),
    ("What is a JavaScript variable?", "Variables in JavaScript are declared with let, const, or var to store data values."),
    ("What is DOM in JavaScript?", "The DOM is the Document Object Model, representing the web page as a tree of objects JavaScript can modify."),
    ("What is Node.js?", "Node.js is a runtime that lets you run JavaScript outside the browser for server-side applications."),
    ("What is JSON?", "JSON is JavaScript Object Notation, a lightweight data format easy for humans and machines to read."),
    ("What are JavaScript promises?", "Promises represent values available later, helping handle asynchronous operations."),
    ("What is the difference between let and const?", "let declares variables that can be reassigned. const declares constants that cannot change."),

    # WEB DEVELOPMENT
    ("What is HTML?", "HTML is HyperText Markup Language. It defines the structure and content of web pages using tags."),
    ("What is CSS?", "CSS is Cascading Style Sheets. It controls how HTML elements look including colors, fonts, and layout."),
    ("What is a web browser?", "A web browser is software that retrieves and displays web pages like Chrome, Firefox, or Safari."),
    ("What is a website?", "A website is a collection of related web pages under one domain."),
    ("What is a web server?", "A web server stores and delivers web pages when your browser requests them."),
    ("What is an API?", "An API is an Application Programming Interface that defines how software components communicate."),
    ("What is REST?", "REST is an architectural style for web APIs using HTTP methods like GET and POST."),
    ("What is a database?", "A database is an organized collection of data stored electronically."),
    ("What is SQL?", "SQL is Structured Query Language for communicating with relational databases."),
    ("What is frontend?", "Frontend is the part of a website users see and interact with, including HTML, CSS, and JavaScript."),
    ("What is backend?", "Backend is the server-side part handling data storage and business logic."),
    ("What is responsive design?", "Responsive design makes websites adapt to different screen sizes automatically."),

    # AI & MACHINE LEARNING
    ("What is AI?", "AI is Artificial Intelligence, computer systems that perform tasks requiring human intelligence like understanding language."),
    ("What is machine learning?", "Machine learning is AI where computers learn from data without explicit programming."),
    ("What is deep learning?", "Deep learning uses neural networks with many layers to learn complex patterns."),
    ("What is a neural network?", "A neural network is a computing system inspired by the brain with interconnected nodes."),
    ("What is NLP?", "NLP is Natural Language Processing, enabling computers to understand human language."),
    ("What is a language model?", "A language model is AI trained to understand and generate text."),
    ("What is training data?", "Training data is data used to teach a machine learning model."),
    ("What is overfitting?", "Overfitting happens when a model learns training data too closely and fails on new data."),
    ("What is underfitting?", "Underfitting occurs when a model is too simple to capture patterns in data."),
    ("What is a feature?", "A feature is an input variable used by a machine learning model."),
    ("What is supervised learning?", "Supervised learning trains models on labeled data with known correct answers."),
    ("What is unsupervised learning?", "Unsupervised learning finds patterns in data without labels."),

    # DATA SCIENCE
    ("What is data analysis?", "Data analysis examines data to find insights and patterns for decision-making."),
    ("What is data science?", "Data science combines programming, statistics, and domain expertise to extract knowledge from data."),
    ("What is a dataset?", "A dataset is a collection of structured data for analysis."),
    ("What is data visualization?", "Data visualization presents data graphically using charts and graphs."),
    ("What is mean median and mode?", "Mean is the average, median is the middle value, and mode is the most frequent value."),
    ("What is standard deviation?", "Standard deviation measures how spread out data values are from the mean."),
    ("What is correlation?", "Correlation measures the relationship between two variables."),
    ("What is big data?", "Big data refers to extremely large datasets requiring special tools to process."),

    # COMPUTER SCIENCE CONCEPTS
    ("What is a CPU?", "CPU is the Central Processing Unit, executing instructions and performing calculations."),
    ("What is RAM?", "RAM is Random Access Memory, temporary storage for data the CPU needs quickly."),
    ("What is a hard drive?", "A hard drive is permanent storage that retains data when the computer is off."),
    ("What is an operating system?", "An operating system manages computer hardware like Windows, macOS, or Linux."),
    ("What is an SSD?", "SSD is a Solid State Drive, faster storage with no moving parts."),
    ("What is a GPU?", "GPU is the Graphics Processing Unit for rendering images and parallel computing."),
    ("What is an IP address?", "An IP address is a unique number identifying devices on a network."),
    ("What is cloud computing?", "Cloud computing delivers computing services over the internet on demand."),
    ("What is encryption?", "Encryption converts data into a secret code to protect it."),
    ("What is open source software?", "Open source software has publicly available code anyone can view and modify."),

    # GIT & VERSION CONTROL
    ("What is Git?", "Git is a version control system tracking changes in code for collaboration."),
    ("What is GitHub?", "GitHub is a platform hosting Git repositories online with collaboration tools."),
    ("What is a repository?", "A repository is storage for code and its complete history."),
    ("What is a commit?", "A commit is a snapshot of changes saved to the repository."),
    ("What is a branch?", "A branch is a parallel version of code for developing features."),
    ("What is a merge?", "Merging combines changes from different branches."),
    ("What is a pull request?", "A pull request proposes changes for review before merging."),

    # LEARNING & HELP
    ("How do I start programming?", "Choose a beginner-friendly language like Python. Learn basics through tutorials and practice regularly."),
    ("What programming language should I learn first?", "Python is often recommended for beginners due to its clean, readable syntax."),
    ("How can I improve at coding?", "Practice consistently, read others' code, build projects, and learn from mistakes."),
    ("Where can I learn to code?", "Try freeCodeCamp, Codecademy, Coursera, or YouTube tutorials."),
    ("How long does it take to learn programming?", "Basics take weeks, mastery takes years. Start simple and gradually build complexity."),
    ("What projects should I build as a beginner?", "Start with a calculator, to-do list, or simple game. Increase complexity as you learn."),
    ("How do I debug my code?", "Read error messages, use print statements to track values, and isolate the problem."),
    ("What is pair programming?", "Pair programming is two developers working together, one writing code while the other reviews."),
    ("My code isn't working. What do I do?", "Read error messages carefully, check line numbers, and test parts separately."),
    ("How do I fix a syntax error?", "Check for missing punctuation like colons, parentheses, or brackets."),
    ("How do I fix a logic error?", "Trace through your code manually with example inputs to find where it differs from expectations."),
    ("What if my program runs but gives wrong output?", "Work backward from the output, tracing variable values to find where they start differing."),
    ("How do I handle errors in my code?", "Use try-except blocks to catch exceptions and provide helpful error messages."),
    ("What is rubber duck debugging?", "Explain your code line-by-line to a rubber duck. Articulating often reveals the problem."),

    # Additional diverse questions for better vocabulary
    ("Can you explain machine learning?", "Machine learning is AI where computers automatically improve through experience. They find patterns in data without being explicitly programmed for each task."),
    ("Tell me about neural networks", "Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes that process information."),
    ("What programming languages exist?", "Many languages exist including Python, JavaScript, Java, C++, Go, Rust, Ruby, PHP, Swift, Kotlin, and TypeScript. Each has strengths for different use cases."),
    ("How does the internet work?", "The internet is a global network of connected computers. Data travels through routers and switches using protocols like TCP/IP to reach destinations."),
    ("Explain cloud computing benefits", "Cloud computing offers scalability, cost efficiency, and accessibility. You only pay for resources used and can scale up or down based on demand."),
    ("What is a framework?", "A framework is pre-written code providing structure for applications. It handles common tasks so developers can focus on unique features."),
    ("What is the difference between frontend and backend?", "Frontend handles user interface and experience. Backend manages servers, databases, and business logic. They work together to create complete applications."),
    ("Tell me about clean code", "Clean code is readable, maintainable, and well-organized. Follow conventions, use meaningful names, write small functions, and add comments only when necessary."),
    ("What is refactoring?", "Refactoring improves code structure without changing behavior. It makes code cleaner and more maintainable while preserving functionality."),
    ("Explain unit testing", "Unit testing verifies individual components work correctly. Write tests for each function to catch bugs early and ensure changes don't break existing code."),
    ("What is version control?", "Version control tracks changes to files over time. It lets you revert to previous states, collaborate with others, and maintain history."),
    ("How do I write good documentation?", "Document the what, why, and how. Explain purpose, parameters, return values, and usage examples. Keep it updated as code changes."),
    ("What are coding best practices?", "Follow consistent naming, write small focused functions, handle errors gracefully, test thoroughly, and refactor regularly."),
    ("Explain technical debt", "Technical debt is the cost of taking shortcuts. Quick fixes may work now but require more work later to properly implement."),
    ("What is continuous integration?", "CI automatically builds and tests code changes. It catches problems early by integrating work frequently and running automated tests."),
    ("Tell me about code review", "Code review is examining others' code before merging. It improves quality, shares knowledge, catches bugs, and ensures adherence to standards."),
    ("What is DevOps?", "DevOps combines development and operations practices. It automates infrastructure, emphasizes collaboration, and speeds up delivery."),
    ("Explain microservices", "Microservices architecture splits applications into small, independent services. Each handles specific functionality and communicates through APIs."),
    ("What is a monorepo?", "A monorepo stores multiple projects in one repository. It simplifies dependency management and enables atomic changes across codebases."),
    ("How do I handle dependencies?", "Use package managers to install and track dependencies. Pin versions for reproducibility, update regularly, and remove unused packages."),
    ("What is technical writing?", "Technical writing communicates complex information clearly. Focus on audience needs, use simple language, and include examples."),
    ("Explain RESTful APIs", "RESTful APIs follow REST principles using HTTP methods. Resources have URLs, and operations use GET, POST, PUT, DELETE appropriately."),
    ("What is GraphQL?", "GraphQL is a query language for APIs. Clients request exactly what they need, avoiding over-fetching or under-fetching data."),
    ("Tell me about authentication", "Authentication verifies user identity through credentials. Common methods include passwords, tokens, biometrics, and multi-factor authentication."),
    ("What is authorization?", "Authorization determines what authenticated users can access. It controls permissions based on roles, attributes, or policies."),
    ("Explain caching", "Caching stores frequently accessed data for quick retrieval. It reduces server load, speeds response times, and improves user experience."),
    ("What are microcontrollers?", "Microcontrollers are compact integrated circuits for embedded systems. They contain a processor, memory, and I/O peripherals."),
    ("Tell me about containerization", "Containerization packages applications with their dependencies. Containers run consistently across environments, making deployment easier."),
    ("What is virtualization?", "Virtualization creates virtual versions of hardware, storage, or networks. It enables multiple systems to run on one physical machine."),
    ("Explain orchestration", "Orchestration automates coordination of multiple systems. Tools like Kubernetes manage containers at scale, handling deployment and scaling."),
    ("What is load balancing?", "Load balancing distributes traffic across multiple servers. It improves reliability, prevents overload, and ensures high availability."),
    ("How do databases work?", "Databases store and organize data for efficient retrieval. They use structures like tables with relationships to manage information."),
    ("Explain indexing in databases", "Indexing creates data structures for faster lookups. Without indexes, databases scan entire tables; indexes enable direct access."),
    ("What is normalization?", "Normalization organizes data to reduce redundancy. It splits information into related tables, ensuring each fact is stored once."),
    ("Tell me about ACID properties", "ACID ensures reliable transactions: Atomicity (all or nothing), Consistency (valid state), Isolation (concurrent independence), Durability (permanent)."),
    ("What is NoSQL?", "NoSQL databases handle unstructured or semi-structured data. Types include document, key-value, column-family, and graph databases."),
    ("How do APIs handle errors?", "APIs return appropriate status codes and error messages. Use 4xx for client errors, 5xx for server errors, with descriptive messages."),
    ("What is rate limiting?", "Rate limiting controls how many requests a client can make. It prevents abuse, ensures fair usage, and protects server resources."),
    ("Explain webhooks", "Webhooks send automatic notifications between applications. Instead of polling, servers push data when events occur."),
    ("What is OAuth?", "OAuth is an authorization framework for secure access. Users grant limited permissions to applications without sharing passwords."),
    ("Tell me about JWT tokens", "JWT is a JSON-based token format for authentication. It contains claims about the user, signed to prevent tampering."),
    ("What is CORS?", "CORS controls which domains can access resources. Browsers enforce it to prevent unauthorized cross-origin requests."),
    ("Explain DNS", "DNS translates domain names to IP addresses. It's the phonebook of the internet, directing requests to correct servers."),
    ("What is SSL/TLS?", "SSL and TLS encrypt data between clients and servers. They ensure privacy, authenticate identity, and protect against tampering."),
    ("How does HTTPS work?", "HTTPS is HTTP secured with TLS. It encrypts communication, verifies server identity, and protects against interception."),
    ("Tell me about clean architecture", "Clean architecture separates concerns into layers. Business logic is independent of frameworks, databases, and UI."),
    ("What is SOLID principles?", "SOLID: Single responsibility, Open for extension, Liskov substitution, Interface segregation, Dependency inversion."),
    ("Explain design patterns", "Design patterns are reusable solutions to common problems. Examples include Singleton, Factory, Observer, and Strategy patterns."),
    ("What is a singleton pattern?", "Singleton ensures a class has only one instance. It provides global access while preventing multiple instantiations."),
    ("Tell me about factory pattern", "Factory creates objects without specifying exact classes. It delegates instantiation to subclasses or methods."),
    ("What is dependency injection?", "Dependency injection passes dependencies to objects. It decouples components, improves testability, and manages dependencies externally."),
    ("How do I structure a project?", "Organize by feature or layer. Keep related code together, separate concerns, and follow consistent conventions."),
    ("What makes code maintainable?", "Clear naming, small functions, minimal coupling, strong cohesion, good documentation, and comprehensive tests."),
    ("Explain technical interviews", "Technical interviews assess coding and problem-solving skills. Practice algorithms, data structures, system design, and communication."),
    ("How do I prepare for coding interviews?", "Practice on platforms like LeetCode. Review data structures, algorithms, and system design. Mock interviews help reduce anxiety."),
    ("What are data structures?", "Data structures organize and store data efficiently. Common types include arrays, linked lists, stacks, queues, trees, and graphs."),
    ("Explain Big O notation", "Big O describes algorithm efficiency. It measures how time or space grows as input increases, helping compare algorithms."),
    ("What is recursion?", "Recursion is when a function calls itself. It breaks problems into smaller subproblems with a base case to stop."),
    ("Tell me about sorting algorithms", "Sorting arranges data in order. Common algorithms include bubble sort, merge sort, quick sort, and insertion sort."),
    ("What is binary search?", "Binary search finds items in sorted arrays efficiently. It repeatedly halves the search space, achieving O(log n) complexity."),
    ("How do graphs work?", "Graphs represent relationships between objects. Nodes connect through edges, enabling modeling of networks, maps, and connections."),
    ("What are trees in computer science?", "Trees are hierarchical data structures with a root, branches, and leaves. They're used in file systems, databases, and parsing."),
    ("Explain linked lists", "Linked lists store elements in nodes connected by pointers. Unlike arrays, they allow efficient insertion at any position."),
    ("What is a stack?", "A stack follows Last-In-First-Out order. Push adds to top, pop removes from top. Used in function calls and parsing."),
    ("Tell me about queues", "Queues follow First-In-First-Out order. Used for task scheduling, buffering, and handling requests in order."),
    ("What is a hash table?", "Hash tables use keys to access values quickly. A hash function maps keys to indices, enabling O(1) average lookups."),
    ("Explain dynamic programming", "Dynamic programming solves problems by breaking them into overlapping subproblems. Store solutions to avoid recomputation."),
    ("What is greedy algorithm?", "Greedy algorithms make locally optimal choices at each step. They're simple but don't always find global optima."),
    ("How do I optimize code?", "Profile to find bottlenecks. Choose better algorithms, reduce unnecessary work, cache results, and avoid premature optimization."),
    ("What is parallel programming?", "Parallel programming executes multiple tasks simultaneously. It utilizes multiple processors or cores to speed up computation."),
    ("Tell me about concurrency", "Concurrency manages multiple tasks at once. Not necessarily simultaneous, but handles overlapping operations safely."),
    ("What are race conditions?", "Race conditions occur when operations overlap unpredictably. Results depend on timing, causing bugs that are hard to reproduce."),
    ("How do I handle thread safety?", "Use locks, semaphores, or atomic operations. Synchronize access to shared resources to prevent race conditions."),
    ("What is deadlock?", "Deadlock happens when threads wait indefinitely for each other. Prevention includes ordering locks and using timeouts."),
    ("Explain memory management", "Memory management allocates and frees memory. Languages handle it differently: automatic garbage collection, manual management, or ownership systems."),
    ("What is garbage collection?", "Garbage collection automatically frees unused memory. It identifies unreachable objects and reclaims their space."),
    ("Tell me about memory leaks", "Memory leaks occur when allocated memory isn't freed. They cause programs to consume increasing memory, eventually crashing."),
    ("What is buffer overflow?", "Buffer overflow writes beyond allocated memory. It can corrupt data, crash programs, or allow security exploits."),
    ("How do I profile applications?", "Use profiling tools to measure CPU, memory, and I/O. Identify hotspots, allocation patterns, and performance bottlenecks."),
    ("What is benchmarking?", "Benchmarking measures performance under controlled conditions. Compare implementations, track regressions, and validate optimizations."),
    ("Explain static analysis", "Static analysis examines code without running it. Tools detect bugs, style issues, and potential problems before execution."),
    ("What is dynamic analysis?", "Dynamic analysis tests running applications. It finds memory errors, race conditions, and runtime behavior issues."),
    ("How do linters help?", "Linters check code for style and potential errors. They enforce standards and catch issues early in development."),
    ("Tell me about formatters", "Formatters automatically style code. They ensure consistent formatting across teams, reducing review friction."),
    ("What is software architecture?", "Software architecture defines high-level structure. It establishes components, relationships, and guiding principles for the system."),
    ("Explain modularity", "Modularity divides systems into independent modules. Each handles specific functionality with clear interfaces to others."),
    ("What is coupling?", "Coupling measures dependency between modules. Low coupling means modules are independent; high coupling makes changes harder."),
    ("Tell me about cohesion", "Cohesion measures how focused a module is. High cohesion means elements work together for one purpose."),
    ("What are common architecture patterns?", "Patterns include layered, hexagonal, microservices, event-driven, and CQRS. Choose based on requirements."),
    ("How do I scale applications?", "Scale vertically with bigger servers or horizontally with more instances. Use caching, load balancing, and database sharding."),
    ("What is horizontal scaling?", "Horizontal scaling adds more machine instances. It improves capacity and fault tolerance through distribution."),
    ("Explain vertical scaling", "Vertical scaling upgrades existing hardware. More CPU, RAM, or storage handles increased load on one machine."),
    ("What is sharding?", "Sharding splits databases across multiple servers. Each shard holds a portion of data, enabling horizontal scaling."),
    ("Tell me about replication", "Replication copies data across multiple servers. It improves availability and read performance through redundancy."),
    ("What is eventual consistency?", "Eventual consistency means updates propagate over time. Systems become consistent eventually, allowing temporary divergence."),
    ("How do message queues work?", "Message queues decouple producers and consumers. Messages wait in queues until processed, enabling async communication."),
    ("Explain pub/sub pattern", "Publish/subscribe has publishers send messages and subscribers receive topics. Subscribers choose what to receive."),
    ("What is idempotence?", "Idempotent operations produce the same result when called multiple times. Important for reliable distributed systems."),
    ("Tell me about distributed systems", "Distributed systems span multiple networked computers. They coordinate to appear as a single coherent system."),
    ("What is CAP theorem?", "CAP theorem states distributed systems can guarantee at most two of: Consistency, Availability, Partition tolerance."),
    ("How do I handle failures?", "Implement retries with backoff, circuit breakers, fallbacks, and graceful degradation. Monitor and alert on failures."),
    ("What is chaos engineering?", "Chaos engineering deliberately introduces failures. It reveals weaknesses before they cause real outages."),
    ("Why is observability important?", "Observability provides visibility into system behavior. Logs, metrics, and traces help debug issues and understand performance."),
]


def generate_variations(user_text, assistant_text):
    """Generate variations of a conversation pair."""
    variations = [(user_text, assistant_text)]

    # Lowercase version
    if user_text.lower() != user_text:
        variations.append((user_text.lower(), assistant_text))

    # Add question mark if missing
    if '?' not in user_text and not user_text.endswith('.'):
        variations.append((user_text + "?", assistant_text))

    # Contraction
    if "What is" in user_text:
        variations.append((user_text.replace("What is", "What's"), assistant_text))

    return variations


def main():
    """Generate training data file."""
    all_conversations = []

    # Generate base conversations and variations
    for user_text, assistant_text in BASE_CONVERSATIONS:
        variations = generate_variations(user_text, assistant_text)
        all_conversations.extend(variations)

    # Calculate statistics
    unique_users = len(set(c[0] for c in all_conversations))
    unique_ais = len(set(c[1] for c in all_conversations))

    print(f"Base conversation pairs: {len(BASE_CONVERSATIONS)}")
    print(f"With variations: {len(all_conversations)}")
    print(f"Unique user prompts: {unique_users}")
    print(f"Unique AI responses: {unique_ais}")

    # Write to file in User/AI format
    import os
    os.makedirs("data", exist_ok=True)

    with open("data/diverse_conversations.txt", 'w', encoding='utf-8') as f:
        for user_text, assistant_text in all_conversations:
            f.write(f"User: {user_text}\n")
            f.write(f"AI: {assistant_text}\n")

    print(f"\nSaved {len(all_conversations)} conversation pairs to data/diverse_conversations.txt")


if __name__ == "__main__":
    main()
