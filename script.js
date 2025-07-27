// script.js
// script.js
document.addEventListener('DOMContentLoaded', () => {
    // Previous background and setup code remains the same

    // Add Experience Data
    const experiences = [
        {
            title: "Software Development Intern – Computer Vision",
            company: "Eviden - Atos",
            dates: "March 2025 - Present",
            description: `
            <ul class="experience-list">
                <li>Contributing to the Innovation Team within the Computer Vision Lab, focused on developing client-oriented applications powered by vision models.</li>
                <li>Enhancing the Multi-Target Multi-Camera Tracking (MTMC) system by improving the Re-Identification (ReID) component; designed and validated a novel approach to boost ReID model performance.</li>
                <li>Integrating the proof-of-concept into Ipsotek’s VISuite, a leading video analytics platform.</li>
                <li>Co-developing a Multicam Annotation Tool, a semi-automated tool that allows users to manually correct ReID model predictions across multiple camera views, facilitating the creation of clean datasets for training, testing, and evaluating models and algorithms.</li>
                <li>Technologies: Python, Pytest, C++, GTest, Git, SonarQube, Github Action</li>
            </ul>
            `,
        },
        {
            title: "Resource Planning Optimizaiton Research Intern",
            company: "Institut Mines Télécom",
            dates: "June - August 2024",
            description: `
            <ul class="experience-list">
                <li>La Chaire de recherche "Digital Twin for industrial production systems."</li>
                <li>Analyzed machine performance and production bottlenecks; developed AI-driven approaches to address an NP-hard optimization problem aimed at minimizing production time.</li>
                <li>Reduced weekly production cycles times by 40% (from 107–140 hours to 65–90 hours)—while maintaining the same level of output.</li>
                <li>Designed and implemented a user interface to enable interaction with the optimization tool.</li>
                <li>Awarded an academic score of 19.25/20 for the internship.</li>
                <li>Technologies: Python, Heuristic Optimization, Flask</li>
            </ul>
            `,
            // description: "Pre-trained and Fine-tuned a Custom OCR Model"
        },
        {
            title: "Research and Development Mission",
            company: "IMT-Mines-Ales - clicNwork",
            dates: "January - April 2024",
            description: `
            <ul class="experience-list">
                <li>Preprocessed, visualized, and analyzed time series data to extract insights and identify trends.</li>
                <li>Developed predictive models to forecast temporary workforce demand based on historical data.</li>
                <li>Implemented statistical models including Exponential Smoothing, ARIMA, and Facebook Prophet,and deep learning models LSTM for time series forecasting.</li>
                <li>Technologies: Python, R, Numpy, Pandas, Matplotlib, Pytorch</li>
            </ul>
            `,
            // description: "Pre-trained and Fine-tuned a Custom OCR Model"
        },
        {
            title: "Intern – ERP Deployment (Dolibarr)",
            company: "EFFIdomus",
            dates: "November – December 2023",
            description: `
            <ul class="experience-list">
                <li>Deployment of Dolibarr ERP within the design office to improve internal workflow efficiency.</li>
                <li>Automated the import of large datasets from Excel to Dolibarr, reducing manual input and ensuring data accuracy.</li>
                <li>Assisted in optimizing business processes by configuring and adapting ERP modules to meet specific operational requirements.</li>
                <li>Technologies: Dolibarr, HTML, Excel</li>
            </ul>
            `,
            // description: "Pre-trained and Fine-tuned a Custom OCR Model"
        },
        {
            title: "Research Member",
            company: "ViLa Laboratory - Air Handwriting Recognition for Khmer Characters",
            dates: "2022 - 2023",
            description: `
            <ul class="experience-list">
                <li>Curated a custom Khmer handwritten character dataset and trained a multi-layer perceptron (MLP) for character classification.</li>
                <li>Integrated OpenCV-based gesture tracking to enable real-time air handwriting input.</li>
                <li>Technologies: Python, OpenCV, Pytorch</li>
            </ul>
            `,
            // description: "Pre-trained and Fine-tuned a Custom OCR Model"
        }
    ];

    // Add Certifications Data
    const certifications = [
        {
            title: "Fundamentals of MCP",
            issuer: "Hugging Face",
            date: "2024",
            link: "https://drive.google.com/file/d/1czVQsVSKyvkJpmyNjsa0I2d861IjdvKk/view?usp=sharing"
        },
        {
            title: "LLM",
            issuer: "Hugging Face",
            date: "2025",
            link: "https://drive.google.com/file/d/1VKH7Enj34OHoI9K27NFKovIxacDqoxSj/view?usp=sharing"
        },
        {
            title: "AI Agents Fundamentals",
            issuer: "Hugging Face",
            date: "2025",
            link: "https://drive.google.com/file/d/1QndLNzEzg7N0CpUxgYWBhqXDPOdXOcQe/view?usp=sharing"
        },
        {
            title: "Advanced Learning Algorithms",
            issuer: "Coursera",
            date: "2024",
            link: "https://www.coursera.org/account/accomplishments/verify/8ALUPZRDWUNY"
        },
        {
            title: "ML: Regression and Classification",
            issuer: "Coursera",
            date: "2024",
            link: "https://www.coursera.org/account/accomplishments/verify/WKVBV7H58XPA"
        }
    ];

    // Populate Experience
    const experienceTimeline = document.querySelector('.experience-timeline');
    experiences.forEach(exp => {
        const div = document.createElement('div');
        div.className = 'experience-item';
        div.innerHTML = `
            <h3>${exp.title}</h3>
            <div class="experience-company">${exp.company}</div>
            <div class="experience-dates">${exp.dates}</div>
            <div class="experience-description">${exp.description}</div>
        `;
        experienceTimeline.appendChild(div);
    });
    // Populate Certifications
    const certsGrid = document.querySelector('.certifications-grid');
    certifications.forEach(cert => {
        const div = document.createElement('div');
        div.className = 'certification-card';
        div.innerHTML = `
            <h3><a href="${cert.link}" target="_blank">${cert.title}</a></h3>
            <p>${cert.issuer}</p>
            <div class="cert-date">${cert.date}</div>
        `;
        certsGrid.appendChild(div);
    });

    // Rest of previous JavaScript code remains the same
});
document.addEventListener('DOMContentLoaded', () => {
    // Animated Background
    const canvas = document.getElementById('techBg');
    const ctx = canvas.getContext('2d');
    
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    class Particle {
        constructor() {
            this.reset();
        }
        
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2;
            this.speed = Math.random() * 0.5 + 0.1;
            this.angle = Math.random() * Math.PI * 2;
        }
        
        update() {
            this.x += Math.cos(this.angle) * this.speed;
            this.y += Math.sin(this.angle) * this.speed;
            
            if (this.x < 0 || this.x > canvas.width || 
                this.y < 0 || this.y > canvas.height) {
                this.reset();
            }
        }
        
        draw() {
            ctx.fillStyle = 'rgba(35,134,54,0.3)';
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    const particles = Array(100).fill().map(() => new Particle());
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        requestAnimationFrame(animate);
    }
    
    resizeCanvas();
    animate();
    window.addEventListener('resize', resizeCanvas);

    // Dynamic Content
    const skills = [
        'Python', 'C/C++', 'Java', 'R', 'PyTorch', 'TensorFlow',  'OpenCV', 'Ultralytics (Yolo)', 'Numpy', 'Pandas', 'Matplotlib',
        'Scikit-Learn', 'Pytest', 'GTest', 'Flash', 'Git', 'Github Actions', 'SonarQube', 'Docker', 'Linux', 'Elastic Search','MySQL',
        'MongDB',
    ];
    const knowledges = [
        'Computer Vision', 'Machine Learning', 'Deep Learning', 'GANs & Diffusion Models', 'LLMs', 'Signal Processing', 'Yolo', 'Reinforcment Learning', 'Deep Reinforcement Learning',
    ]

    const projects = [
         {
            id: 'object-detection',
            title: 'Analyzing the Impact of Low-Rank Adaptation for Cross-Domain Few-Shot Object Detection in Aerial Images',
            description: `
                <ul>
                    <li>The research addresses the challenge of cross-domain object detection in few-shot learning. Experiments were conducted using the state-of-the-art object detection model DiffusionDet by applying Parameter-Efficient Fine-Tuning (PEFT), specifically Low-Rank Adaptation (LoRA), to enable cross-domain object detection from everyday images in the COCO dataset to aerial images in the DOTA and DIOR datasets.</li>
                    <li>Personal Contribution: Analyzed the DOTA and DIOR datasets to provide insights into their characteristics and distribution. Explored foundation models including Vision Transformer (ViT), Detection Transformer (DETR), and DiffusionDet, and proposed specific modules/layers for the injection of trainable parameters using LoRA. Monitored gradients during backpropagation throughout the training process to ensure the model was learning effectively, avoiding issues such as gradient explosion or vanishing.</li>
                    <li>Link to paper: <a href="https://arxiv.org/abs/2504.06330">https://arxiv.org/abs/2504.06330</a></li>
                </ul>
            `,
            tech: 'Python, PyTorch, Transformer, LoRA, Git'
        },
        {
            id: 'real-time detection',
            title: 'Détection Automatique des Traînées Astronomiques avec YOLO – Une Approche Exploratoire pour la Connaissance du Domaine Spatial',
            description: `
                <ul>
                    <li>The research addresses the challenge of cross-domain object detection in few-shot learning. The research conducted the experiment on state of the art object detection DiffusionDet by applying PEFT, specifically Low-Rank Adaptation into transforming the images in daily life COCO dataset to aerial images known as DOTA/DIOR.</li>
                    <li>Contribution: Analyzing the DOTA/DIOR dataset to provide insight of their caracteristics and distribution, exploring foudnation models: Vision Transformer (ViT), Detection Transformer (DETR), Diffusion Detection (DiffusionDet) and propose which modules/layers to inject trainable parameters via LoRA, observing the gradient during backpropagation in the learning process to have a general idea wether the model is learning properly.</li>
                    <li>Link to paper: <a href="https://arxiv.org/abs/2504.06330">https://arxiv.org/abs/2504.06330</a></li>
                </ul>
            `,
            tech: 'Python, PyTorch, Transformer, LoRA, Git'
        },
        {
            id: 'academic-chatbot',
            title: 'Academic Chatbot',
            description: `
                <ul class="">
                    <li>Conducted a comparative study of two chatbot development approaches: fine-tuning and Retrieval-Augmented Generation (RAG).</li>
                    <li>Fine-tuned the pre-trained Llama 3.1 8B language model using UnSloth and LoRA, specifically optimized to assist international students by leveraging a custom academic dataset.</li>
                    <li>Developed and implemented a robust RAG system that combines the Llama 3.1 8B model with a hybrid retrieval mechanism using BM25 and embedding model with reranker(BGE-M3 with BGE-Reranker).</li>
                    <li>The dataset included comprehensive information such as:
                        <ul class="sublist">
                            <li>Housing and rental options for students.</li>
                            <li>Study and work conditions for international learners.</li>
                            <li>Detailed descriptions of courses, curricula, and academic requirements in the Department of Informatics and Artificial Intelligence.</li>
                        </ul>
                    </li>
                    <video controls style="width: 100%; height: auto; margin-top: 10px;">
                    Demo
                    <source src="https://github.com/user-attachments/assets/3ba61ac0-f84b-45bf-babd-4988ba909a1e">
                    </video>
                </ul>

            `,
            tech: 'Python, LLM, RAG, PyTorch, Deep Learning, Fine Tuning, LangChain, PERT, Git'
        },
        {
            id: 'object-detection',
            title: 'Application of the LoRA on Object Detection model',
            description: `
                <ul>
                    <li>This research project was conducted in collaboration with a PhD student from Sorbonne University. The objective was to explore the application of LoRA (Low-Rank Adaptation) to enhance the performance of DiffusionDet, an object detection model, under few-shot learning conditions. The goal was to evaluate whether LoRA can effectively adapt the model to new object categories using limited annotated data, improving efficiency and generalization in low-resource scenarios.</li>
                    <li>We applied this approach to aerial imagery datasets, specifically DOTA (Dataset for Object Detection in Aerial Images) and DIOR (Dataset for Object Recognition in Aerial Images). These datasets contain complex scenes with multiple object classes captured from aerial perspectives, making them ideal for evaluating few-shot object detection in real-world scenarios.
                    </li>
                    <li>The paper can be found here: <a href="https://arxiv.org/abs/2504.06330">https://arxiv.org/abs/2504.06330</a></li>
                </ul>
            `,
            tech: 'Python, PyTorch, Transformer, LoRA, Git'
        },
        {
            id: 'ocr',
            title: 'Handwritten Text Recognition with Fine-Tuned TrOCR',
            description: `
                <ul>
                    <li>This project fine-tunes the microsoft/trocr-large-handwritten model from Hugging Face for handwritten text recognition on a custom dataset. The goal is to adapt the model to recognize specific handwritten text styles or domains with improved accuracy.
                    </li>
                    <video controls style="width: 100%; height: auto; margin-top: 10px;">
                    Demo
                    <source src="https://github.com/user-attachments/assets/9c08b124-290e-45eb-88f7-775e8118e7c6">
                    </video>
                </ul>
            `,
            tech: 'Python, PyTorch, Transformer, LoRA, Git'
        },
        {
            id: 'object-classification',
            title: 'Object classification',
            description: `<ul>
                <li>
                    Trained a MobileViT model on the SPOTS-10: Animal Pattern dataset to classify animals based on their body patterns and textures, and on the CIFAR-10 dataset to classify objects. Developed an API using FastAPI, deployed the model trained on CIFAR-10 with Docker on Google Cloud Platform, and created a user interface hosted on Streamlit Community Cloud.
                </li>
                
                    <video controls style="width: 100%; height: auto; margin-top: 10px;">
                    Demo
                    <source src="https://github.com/user-attachments/assets/ae1c0d9a-5c9c-4c3d-885f-fecacd865b87">
                    </video>
            </ul>`,
            
            
            tech: 'Python, PyTorch, Git, Google Cloud Platform, FastAPI, Docker, Streamlit'
        },
        {
            id: 'asr',
            title: 'Automatic Speech Recognition (ASR)',
            description: `
                <ul class="">
                    <li>Developed and fine-tuned the Whisper Automatic Speech Recognition (ASR) model using the PolyAI/minds14 dataset to enhance performance for speech-to-text applications.</li>
                </ul>
            `,
            tech: 'Python, PyTorch, Deep Learning, Fine Tuning, Transformers'
        },
        {
            id: 'handmotion-prediction',
            title: 'HandMotion Prediction',
            description: 'Developing a machine learning project to predict the acceleration of patients hands, providing one prediction per second for each hand from 7 AM to 7 PM. The dataset contains acceleration values (x, y, z) for both hands and corresponding timestamps, with 50 data points recorded per second. Responsibilities include synchronizing datasets for both hands by aligning timestamps, handling missing data through linear interpolation, and training a Long Short-Term Memory (LSTM) model to produce accurate, second-level predictions.',
            tech: 'Python, Data Analysis, PyTorch, Time-series, Git'
        },
        {
            id: 'e-commerce',
            title: 'E-Commerce Web application',
            description: `Developed a web application for sports shoe sales using a Vue.js frontend and a Laravel backend. The frontend, built with Vue.js, provided a dynamic and responsive user interface, enabling users to browse products, filter by size, brand, and category, and manage their shopping cart in real time. The backend, powered by Laravel, handled key functionalities such as user authentication, product management, inventory tracking, order processing, and payment integration. The system also included an admin dashboard for managing product listings, viewing sales analytics, and handling customer queries. The application aimed to deliver a seamless e-commerce experience optimized for both desktop and mobile users.`,
            tech: 'Java script, PHP, Vue.js, Laravel, HTML, CSS, Tailwind, Git'
        },
        {
            id: 'cafe-system',
            title: 'Café System',
            description: `Developed a Point of Sale (POS) management system for a café using Java and the Spring Boot backend framework. The system was designed to streamline daily operations by managing orders, inventory, tables, billing, and staff roles. Key features included real-time order tracking, digital receipts, menu management, and role-based access for cashiers and administrators. The backend was built with Spring Boot to ensure scalability, modularity, and efficient API handling. The application aimed to improve service speed, reduce human error, and provide insightful sales reports for business analysis.`,
            tech: 'Java, Spring Boot, HTML, CSS, Tailwind, Java script ,Git'
        },
        {
            id: 'air-writing',
            title: 'Air Writing',
            description: `Collected and prepared khmer dataset to train and fine-tune an existing machine learning model with the goal of improving its prediction accuracy. The process involved data cleaning, augmentation, and labeling to ensure high-quality input. The model was retrained using the updated dataset, followed by performance evaluation through metrics such as accuracy, precision, recall, and F1-score. This iterative process helped enhance the model’s generalization and reliability in real-world scenarios.`,
            tech: 'Python, PyTorch ,Git'
        },
    ];

    

    // Populate Skills
    const skillsGrid = document.querySelector('.skills-grid');
    skills.forEach(skill => {
        const div = document.createElement('div');
        div.className = 'skill-card';
        div.textContent = skill;
        skillsGrid.appendChild(div);
    });

    // Populate knowledges
    const knowledgesGrid = document.querySelector('.skills-grid');
    knowledges.forEach(knowledge => {
        const div = document.createElement('div');
        div.className = 'skills-grid';
        div.textContent = knowledge;
        knowledgesGrid.appendChild(div);
    });

    // Populate Projects
    const projectsGrid = document.querySelector('.projects-grid');
    projects.forEach(project => {
        const div = document.createElement('div');
        div.className = 'project-card';
        div.id = project.id;
        div.innerHTML = `
            <h3>${project.title}</h3>
            <p>${project.description}</p>
            <div class="tech">${project.tech}</div>
        `;
        projectsGrid.appendChild(div);
    });

    // Check for anchor link on page load
    window.addEventListener('DOMContentLoaded', () => {
        const hash = window.location.hash.substring(1); // Get #id from URL
        if (hash) {
            const project = document.getElementById(hash);
            if (project) {
                project.scrollIntoView({ behavior: 'smooth' });
                project.style.backgroundColor = '#f8f9fa'; // Optional highlight
            }
        }
    });

    // Mobile Menu
    const menuBtn = document.querySelector('.menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    menuBtn.addEventListener('click', () => {
        navLinks.style.display = navLinks.style.display === 'flex' ? 'none' : 'flex';
    });

    // Smooth Scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});

// script.js
// Add education data
const education = [
    {
        degree: "MSc in Artificial Intelligence and Data Science",
        institution: "IMT Mines Ales",
        dates: "2023 - 2025",
        details: ""
    },
    {
        degree: "BSc in Computer Science",
        institution: "Institute of Technology of Cambodia",
        dates: "2019 - 2023",
        details: ""
    }
];

// Populate Education
const educationTimeline = document.querySelector('.education-timeline');
education.forEach(edu => {
    const div = document.createElement('div');
    div.className = 'education-item';
    div.innerHTML = `
        <h3>${edu.degree}</h3>
        <div class="education-institution">${edu.institution}</div>
        <div class="education-dates">${edu.dates}</div>
        <p class="education-details">${edu.details}</p>
    `;
    educationTimeline.appendChild(div);
});