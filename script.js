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
                <li>An academic score of 19.25/20 for the internship.</li>
                <li>Technologies: Python, Heuristic Optimization, Flask, Pandas, Numpy</li>
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
                <li>Curated a custom Khmer handwritten character dataset and trained a multi-layer perceptron (MLP) for Khmer character classification.</li>
                <li>Integrated OpenCV-based gesture tracking to enable real-time air handwriting input.</li>
                <li>Achieved 89.06% accuracy.</li>
                <li>Technologies: Python, OpenCV, Pytorch</li>
            </ul>
            `,
            // description: "Pre-trained and Fine-tuned a Custom OCR Model"
        }
    ];

    // Add Certifications Data
    const certifications = [
        {
            title: "GANs: Complete Guide",
            issuer: "Udemy - AI Expert Academy",
            date: "2025",
            link: "https://drive.google.com/file/d/18FDgbh3TNkflbgSnnLaqDhv209wEMCMj/view?usp=sharing"
        },
        {
            title: "Data Analysis",
            issuer: "Coursera - Google",
            date: "2024",
            link: "https://drive.google.com/file/d/1VJ1AuR_YR98rsmg8p68Av6J4zWTolwDy/view?usp=sharing"
        },
        
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
        'Scikit-Learn', 'Pytest', 'GTest', 'Flask', 'FastAPI', 'Git', 'Github Actions', 'SonarQube', 'Docker', 'Linux', 'Elastic Search','MySQL',
        'MongDB',
    ];
    const knowledges = [
        'Computer Vision', 'Machine Learning', 'Deep Learning', 'GANs & Diffusion Models', 'LLMs', 'Signal Processing', 'Yolo', 'Deep Reinforcement Learning', 'Software Development'
    ]

    const projects = [
         {
            id: 'object-detection',
            title: 'Conference EUSIPCO 2025: Analyzing the Impact of Low-Rank Adaptation for Cross-Domain Few-Shot Object Detection in Aerial Images',
            description: `
                <ul>
                    <li>The research addresses the challenge of cross-domain object detection in few-shot learning. Experiments were conducted using the state-of-the-art object detection model DiffusionDet by applying Parameter-Efficient Fine-Tuning (PEFT), specifically Low-Rank Adaptation (LoRA), to enable cross-domain object detection from everyday images in the COCO dataset to aerial images in the DOTA and DIOR datasets.</li>
                    <li>Personal Contribution: Analyzed the DOTA and DIOR datasets to provide insights into their characteristics and distribution. Explored foundation models including Vision Transformer (ViT), Detection Transformer (DETR), and DiffusionDet, and proposed specific modules/layers for the injection of trainable parameters using LoRA. Monitored gradients during backpropagation throughout the training process to ensure the model was learning properly, avoiding issues such as gradient explosion or vanishing.</li>
                    <li>Link to paper: <a href="https://arxiv.org/abs/2504.06330">https://arxiv.org/abs/2504.06330</a></li>
                </ul>
            `,
            tech: 'Python, PyTorch, Transformer, LoRA, Git'
        },
        {
            id: 'real-time detection',
            title: 'Conference APIA 2025: Détection Automatique des Traînées Astronomiques avec YOLO – Une Approche Exploratoire pour la Connaissance du Domaine Spatial',
            description: `
                <ul>
                    <li>A YOLOv8 model was fine-tuned from daily image data (COCO) to telescope images collected in the Luxembourg region to detect streaks from satellites and space debris. The model was optimized for varying lighting and noise conditions. It reached 0.90 mAP@50-95 and processed at 91 fps, supporting real-time deployment on edge devices. Further optimization can be applied to manage the trade-off between speed and accuracy.</li>
                    <li>Link to paper: <a href="https://pfia2025.u-bourgogne.fr/conferences/apia/Articles/D%C3%A9tection%20Automatique%20des%20Tra%C3%AEn%C3%A9es%20Astronomiques%20avec%20YOLO%20-%20Une%20Approche%20Exploratoire%20pour%20la%20Connaissance%20du%20Domaine%20Spatial.pdf">APIA 2025</a></li>
                </ul>
            `,
            tech: 'Python, OpenCV, Ultralytics (Yolo)'
        },
        {
            id: 'hackathon',
            title: 'Hackathon 1st place: Patient Activity Prediction from Bracelet Acceleration Signals',
            description: `
                <ul>
                    <li>
                        The main objective of the hackathon was to predict patient activities based on data collected from two bracelets worn on each hand. The dataset contained Cartesian coordinates (x, y, z) for both hands, representing activities such as eating, sleeping, bathing, and changing clothes. The signal was highly noisy and included missing values. Various preprocessing techniques were explored to retain only meaningful semantic information for accurate prediction. Methods such as interpolation were applied to handle missing data. As a team of five, we experimented with a range of models—from traditional machine learning approaches to deep learning and transformer-based architectures.
                    </li>
                    <li>
                        A key aspect of the project involved transforming the signal from the time domain to the frequency domain using the Fast Fourier Transform (FFT), followed by applying Linear Discriminant Analysis (LDA) to reduce the frequency-domain features to a 2D space. This enabled us to visualize class separation. While certain activities were clearly distinguishable, others overlapped significantly, suggesting that acceleration data alone (x, y, z) may not be sufficient to differentiate between similar types of activities.
                    </li>
                </ul>


            `,
            tech: 'Python, Pandas, Numpy, FFT, LDA, Transformer, LSTM'
        },
        {
            id: 'llm-rag-lora',
            title: 'Comparing RAG and LoRA for Metaphor Comprehension: A Study on Mistral-7B with the LCC Dataset',
            description: `
                <ul>
                    <li>
                        LLMs—particularly smaller models like Mistral 7B—often struggle with metaphor comprehension due to limited contextual reasoning. 
                        To address this, metaphor understanding was evaluated using the LCC Metaphor Dataset via two approaches: 
                        <strong>LoRA fine-tuning</strong> and <strong>Retrieval-Augmented Generation (RAG) using ChromaDB</strong>. 
                        The objective was to predict the target domain of a metaphor given the full metaphorical phrase. 
                        Evaluation was based on prediction accuracy, allowing for minor variations in phrasing.
                    </li>
                    <li>
                        <strong>RAG achieved 99.8% accuracy</strong>, significantly outperforming the LoRA-tuned model (38.1%) and the baseline (34.9%). 
                        The results demonstrate that RAG is highly effective for structured, low-resource tasks, while LoRA offers a scalable and lightweight alternative.
                    </li>
                </ul>
            `,
            tech: 'Python, Pytorch, Transformer, Low-rank Adaptation (LoRA), Retrieval-Augmented Generation (RAG), ChromaDB'
        },
        // {
        //     id: 'ocr',
        //     title: 'Handwritten Text Recognition with Fine-Tuned TrOCR',
        //     description: `
        //         <ul>
        //             <li>This project fine-tunes the microsoft/trocr-large-handwritten model from Hugging Face for handwritten text recognition on a custom dataset. The goal is to adapt the model to recognize specific handwritten text styles or domains with improved accuracy.
        //             </li>
        //             <video controls style="width: 100%; height: auto; margin-top: 10px;">
        //             Demo
        //             <source src="https://github.com/user-attachments/assets/9c08b124-290e-45eb-88f7-775e8118e7c6">
        //             </video>
        //         </ul>
        //     `,
        //     tech: 'Python, PyTorch, Transformer, LoRA, Git'
        // },
        // {
        //     id: 'object-classification',
        //     title: 'Object classification',
        //     description: `<ul>
        //         <li>
        //             Trained a MobileViT model on the SPOTS-10: Animal Pattern dataset to classify animals based on their body patterns and textures, and on the CIFAR-10 dataset to classify objects. Developed an API using FastAPI, deployed the model trained on CIFAR-10 with Docker on Google Cloud Platform, and created a user interface hosted on Streamlit Community Cloud.
        //         </li>
                
        //             <video controls style="width: 100%; height: auto; margin-top: 10px;">
        //             Demo
        //             <source src="https://github.com/user-attachments/assets/ae1c0d9a-5c9c-4c3d-885f-fecacd865b87">
        //             </video>
        //     </ul>`,
            
            
        //     tech: 'Python, PyTorch, Git, Google Cloud Platform, FastAPI, Docker, Streamlit'
        // },
        // {
        //     id: 'asr',
        //     title: 'Automatic Speech Recognition (ASR)',
        //     description: `
        //         <ul class="">
        //             <li>Developed and fine-tuned the Whisper Automatic Speech Recognition (ASR) model using the PolyAI/minds14 dataset to enhance performance for speech-to-text applications.</li>
        //         </ul>
        //     `,
        //     tech: 'Python, PyTorch, Deep Learning, Fine Tuning, Transformers'
        // },
        // {
        //     id: 'handmotion-prediction',
        //     title: 'HandMotion Prediction',
        //     description: 'Developing a machine learning project to predict the acceleration of patients hands, providing one prediction per second for each hand from 7 AM to 7 PM. The dataset contains acceleration values (x, y, z) for both hands and corresponding timestamps, with 50 data points recorded per second. Responsibilities include synchronizing datasets for both hands by aligning timestamps, handling missing data through linear interpolation, and training a Long Short-Term Memory (LSTM) model to produce accurate, second-level predictions.',
        //     tech: 'Python, Data Analysis, PyTorch, Time-series, Git'
        // },
        // {
        //     id: 'e-commerce',
        //     title: 'E-Commerce Web application',
        //     description: `Developed a web application for sports shoe sales using a Vue.js frontend and a Laravel backend. The frontend, built with Vue.js, provided a dynamic and responsive user interface, enabling users to browse products, filter by size, brand, and category, and manage their shopping cart in real time. The backend, powered by Laravel, handled key functionalities such as user authentication, product management, inventory tracking, order processing, and payment integration. The system also included an admin dashboard for managing product listings, viewing sales analytics, and handling customer queries. The application aimed to deliver a seamless e-commerce experience optimized for both desktop and mobile users.`,
        //     tech: 'Java script, PHP, Vue.js, Laravel, HTML, CSS, Tailwind, Git'
        // },
        {
            id: 'ai-web-app',
            title: 'Multimodal AI Web Application',
            description: `
                <ul>
                    <li>Developed a Flask-based web application integrating vision and language models.</li>
                    <li>Implemented image classification using a pretrained Vision Transformer (ViT) model from Hugging Face.</li>
                    <li>Integrated real-time object detection with YOLOv3 via OpenCV and webcam streaming.</li>
                    <li>Built a chatbot interface powered by a lightweight LLM for natural language conversation.</li>
                    <li>Enabled session-based conversation history and response generation via a custom text-generation module.</li>
                    <li>Designed multiple endpoints for image upload, classification, and chat interaction with dynamic rendering in HTML.</li>
                </ul>
            `,
            tech: 'Python, Flask, OpenCV, PyTorch, Transformers, HTML, CSS, JavaScript'
        },
        {
            id: 'robot-arm-rl',
            title: 'Musculoskeletal Control of a Robot Arm using Reinforcement Learning',
            description: `
                <ul>
                <li>Designed a custom musculoskeletal environment simulating a two-joint arm with 11 muscles, incorporating biomechanical properties such as inertia, damping, and gravity. The task was modeled as a Markov Decision Process with a reward function promoting precision, smoothness, and energy efficiency.</li>
                <li>Trained a reinforcement learning agent using the Soft Actor-Critic (SAC) algorithm to generate muscle activations that allow the arm to reach dynamic targets. The agent achieved stable convergence and generalization across target positions within 250,000 training steps.</li>
                </ul>
            `,
            tech: 'Python, PyTorch, OpenAI Gym, Reinforcement Learning, Soft Actor-Critic (SAC), Biomechanics Simulation'
            }


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
    const knowledgesGrid = document.querySelector('.knowledges-grid');
    knowledges.forEach(knowledge => {
        const div = document.createElement('div');
        div.className = 'knowledges-card';
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
        degree: "BSc in Computer Science and Networks",
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

function setLanguage(lang) {
    document.querySelectorAll('.lang').forEach(el => el.style.display = 'none');

    document.querySelectorAll(`.lang-${lang}`).forEach(el => el.style.display = '');

    document.querySelectorAll('.lang-btn').forEach(btn => btn.classList.remove('active-lang'));
    const activeBtn = document.getElementById(`lang-${lang}`);
    if (activeBtn) activeBtn.classList.add('active-lang');

    localStorage.setItem('preferredLanguage', lang);
}

document.addEventListener('DOMContentLoaded', () => {
    const savedLang = localStorage.getItem('preferredLanguage') || 'fr';
    setLanguage(savedLang);
});

