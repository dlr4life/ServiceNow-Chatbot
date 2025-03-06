// Global variables
let chatVisible = false;

var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', function() {
    try {
        socket.emit('my event', {data: 'User Connected'});
    } catch (error) {
        console.error('Socket connection error:', error);
    }
});

function handleKeyPress(e) {
    if(e.key === 'Enter') {
        sendMessage();
        e.preventDefault();
    }
}

function toggleChat() {
    const modal = document.getElementById('chatModal');
    chatVisible = !chatVisible;
    
    if(chatVisible) {
        modal.classList.add('active');
        document.getElementById('user-input').focus();
        // Only show welcome message if empty
        if(document.getElementById('messages-container').children.length === 0) {
            appendMessage("Welcome! How can I help you today?", false);
        }
    } else {
        modal.classList.remove('active');
    }
}

// Modified sendMessage to handle modal
async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    
    if (message) {
        // Clear input
        userInput.value = '';
        
        // Append user message
        appendMessage(message, true);
        
        try {
            // Get CSRF token before making requests
            fetch('/get-csrf-token')
                .then(response => response.json())
                .then(data => {
                    const response = fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-CSRFToken': data.csrf_token
                        },
                        body: JSON.stringify({ message })
                    });
                    
                    response.then(async (res) => {
                        const data = await res.json();
                        
                        // Append bot response with knowledge articles and catalog items
                        appendMessage(data, false);
                    });
                });
        } catch (error) {
            console.error('Error:', error);
            appendMessage('Sorry, there was an error processing your request.', false);
        }
    }
}

// Improved appendMessage with timestamp
function appendMessage(message, isUser = true) {
    const messagesContainer = document.getElementById('messages-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    if (isUser) {
        messageDiv.textContent = message;
    } else {
        // Handle bot response with knowledge articles and catalog items
        if (typeof message === 'object') {
            const responseDiv = document.createElement('div');
            responseDiv.className = 'bot-response';
            
            // Add main response
            const mainResponse = document.createElement('div');
            mainResponse.className = 'main-response';
            mainResponse.textContent = message.reply;
            responseDiv.appendChild(mainResponse);
            
            // Add knowledge articles if available
            if (message.knowledge_articles && message.knowledge_articles.length > 0) {
                const articlesDiv = document.createElement('div');
                articlesDiv.className = 'knowledge-articles';
                articlesDiv.innerHTML = '<h4>Related Knowledge Articles:</h4>';
                const articlesList = document.createElement('ul');
                message.knowledge_articles.forEach(article => {
                    const li = document.createElement('li');
                    li.textContent = article;
                    articlesList.appendChild(li);
                });
                articlesDiv.appendChild(articlesList);
                responseDiv.appendChild(articlesDiv);
            }
            
            // Add catalog items if available
            if (message.catalog_items && message.catalog_items.length > 0) {
                const catalogDiv = document.createElement('div');
                catalogDiv.className = 'catalog-items';
                catalogDiv.innerHTML = '<h4>Available Services:</h4>';
                const catalogList = document.createElement('ul');
                message.catalog_items.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item;
                    catalogList.appendChild(li);
                });
                catalogDiv.appendChild(catalogList);
                responseDiv.appendChild(catalogDiv);
            }
            
            messageDiv.appendChild(responseDiv);
        } else {
            messageDiv.textContent = message;
        }
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

socket.on('response', function(data) {
    updateChatbox(data.reply);
});

// Add check for drop-zone element
const dropZone = document.getElementById("drop-zone");
if (dropZone) {
    dropZone.addEventListener("drop", function(event) {
        event.preventDefault();
        let data = event.dataTransfer.getData("text");
        document.getElementById("user-input").value = data;

        // Prevent default behavior
        event.preventDefault();
        if (event.dataTransfer.items) {
            // Use DataTransferItemList interface to access the file(s)
            for (var i = 0; i < event.dataTransfer.items.length; i++) {
                if (event.dataTransfer.items[i].kind === 'file') {
                    var file = event.dataTransfer.items[i].getAsFile();
                    if (file.type === "text/plain") {
                        readFileContent(file);
                    }
                }
            }
        }
    });
}

function readFileContent(file) {
    var reader = new FileReader();
    reader.onload = function(event) {
        document.getElementById("user-input").value = event.target.result;
    };
    reader.readAsText(file);
}

function submitFeedback(value) {
    fetch('/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({feedback: value})
    })
    .then(response => {
        if(response.ok) {
            console.log('Feedback submitted');
        }
    })
    .catch(error => console.error('Error:', error));
}

function showHelpModal() {
    const helpModal = new bootstrap.Modal(document.getElementById('helpModal'), {
        backdrop: true,
        keyboard: true,
        focus: true
    });
    helpModal.show();
}

function showSettings() {
    const settingsAccordion = document.getElementById('settingsAccordion');
    if (settingsAccordion) {
        const settingsCollapse = new bootstrap.Collapse(settingsAccordion.querySelector('.accordion-collapse'));
        settingsCollapse.show();
    } else {
        window.location.href = '/#settings';
    }
}

function showSettingsModal() {
    document.getElementById('settingsModal').style.display = 'block';
}

function closeSettingsModal() {
    document.getElementById('settingsModal').style.display = 'none';
}

function updateChatbox(message) {
    const chatbox = document.getElementById("chatbox");
    if (chatbox) {
        chatbox.innerHTML += `<p>${message}</p>`;
        chatbox.scrollTop = chatbox.scrollHeight; // Auto-scroll to the latest message
    }
}

function sendSettings() {
    const sessionTimeout = document.getElementById('sessionTimeout');
    const entityRecognition = document.getElementById('entityRecognition');
    const sentimentAnalysis = document.getElementById('sentimentAnalysis');
    const responseStyle = document.getElementById('responseStyle');
    const confidenceThreshold = document.getElementById('confidenceThreshold');
    const languageSupport = document.getElementById('languageSupport');
    const memoryDuration = document.getElementById('memoryDuration');
    const autoSummarization = document.getElementById('autoSummarization');
    const profanityFilter = document.getElementById('profanityFilter');
    const contextMemory = document.getElementById('contextMemory');

    if (!sessionTimeout || !entityRecognition || !sentimentAnalysis || !responseStyle || 
        !confidenceThreshold || !languageSupport || !memoryDuration || !autoSummarization || 
        !profanityFilter || !contextMemory) {
        console.warn('Some settings elements not found');
        return;
    }

    const settings = {
        sessionTimeout: sessionTimeout.value,
        entityRecognition: entityRecognition.checked,
        sentimentAnalysis: sentimentAnalysis.checked,
        responseStyle: responseStyle.value,
        confidenceThreshold: parseFloat(confidenceThreshold.value),
        languageSupport: Array.from(languageSupport.selectedOptions).map(option => option.value),
        memoryDuration: memoryDuration.value,
        autoSummarization: autoSummarization.checked,
        profanityFilter: profanityFilter.checked,
        contextMemory: contextMemory.checked
    };

    fetch('/save_settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(settings)
    }).then(response => {
        if(response.ok) closeSettingsModal();
    });
}

// Add existence check for confidenceThreshold element
const confidenceThresholdInput = document.getElementById('confidenceThreshold');
if (confidenceThresholdInput) {
    confidenceThresholdInput.addEventListener('input', function(e) {
        const thresholdValue = document.getElementById('thresholdValue');
        if (thresholdValue) {
            thresholdValue.textContent = e.target.value;
        }
    });
}

function sendQuickReply(reply) {
    const inputField = document.getElementById('user-input');
    inputField.value = reply; // Set the input field to the quick reply text
    sendMessage(); // Call the sendMessage function to send the reply
}

function screenshotChat() {
    try {
        const container = document.getElementById("messages-container");
        const messages = container.querySelectorAll('.message');
        let screenshot = '';
        
        messages.forEach((msg) => {
            const isUser = msg.classList.contains('user-message');
            const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            // Get message content
            let content = '';
            if (msg.querySelector('.bot-response')) {
                const mainResponse = msg.querySelector('.main-response');
                const articles = msg.querySelector('.knowledge-articles');
                const catalog = msg.querySelector('.catalog-items');
                
                content = mainResponse ? mainResponse.textContent : '';
                if (articles) {
                    content += '\nKnowledge Articles: ' + Array.from(articles.querySelectorAll('li'))
                        .map(li => li.textContent).join(', ');
                }
                if (catalog) {
                    content += '\nServices: ' + Array.from(catalog.querySelectorAll('li'))
                        .map(li => li.textContent).join(', ');
                }
            } else {
                content = msg.textContent;
            }
            
            screenshot += `${isUser ? 'User' : 'Bot'}: ${content}\n`;
            screenshot += `Time: ${timestamp}\n\n`;
        });
        
        saveScreenshot(screenshot);
    } catch (error) {
        console.error('Failed to capture screenshot:', error);
        alert('Error capturing chat transcript. Please try again.');
    }
}

function saveScreenshot(screenshotText) {
    try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const now = new Date();
        
        // Canvas setup
        const lineHeight = 24;
        const padding = 40;
        const lines = screenshotText.split('\n');
        const maxWidth = 800;
        
        // Calculate dynamic height
        canvas.width = maxWidth;
        canvas.height = Math.max(600, lines.length * lineHeight + padding * 2 + 100);
        
        // Background styling
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#f8f9fa');
        gradient.addColorStop(1, '#e9ecef');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Add decorative elements
        ctx.fillStyle = 'rgba(140, 25, 101, 0.1)';
        ctx.fillRect(0, 0, 8, canvas.height);
        
        // Header section
        ctx.fillStyle = '#8c1965';
        ctx.fillRect(0, 0, canvas.width, 60);
        
        // Header text
        ctx.fillStyle = 'white';
        ctx.font = 'bold 18px "Segoe UI", sans-serif';
        ctx.fillText('Chat Transcript', 20, 35);
        ctx.font = '14px "Segoe UI", sans-serif';
        ctx.fillText(now.toLocaleDateString('en-US', { 
            year: 'numeric', month: 'long', day: 'numeric' 
        }), canvas.width - 200, 35);

        // Content area
        let yPosition = 100;
        ctx.font = '16px "Segoe UI", sans-serif';
        ctx.fillStyle = '#212529';
        ctx.shadowColor = 'rgba(0,0,0,0.05)';
        ctx.shadowBlur = 4;
        ctx.shadowOffsetY = 2;

        // Message styling
        lines.forEach((line, index) => {
            if (line.startsWith('User:') || line.startsWith('Bot:')) {
                // Message bubble
                const isUser = line.startsWith('User:');
                ctx.fillStyle = isUser ? '#8c1965' : '#495057';
                ctx.beginPath();
                ctx.roundRect(50, yPosition - 5, maxWidth - 100, 30, 15);
                ctx.fill();
                
                // Message text
                ctx.fillStyle = 'white';
                ctx.fillText(line.replace(/^(User|Bot):\s*/, ''), 70, yPosition + 15);
            } else if (line.startsWith('Time:')) {
                // Timestamp styling
                ctx.fillStyle = '#6c757d';
                ctx.font = 'italic 12px "Segoe UI", sans-serif';
                ctx.fillText(line.replace('Time: ', ''), maxWidth - 150, yPosition - 10);
                yPosition += 10;
            } else if (line.startsWith('Knowledge Articles:') || line.startsWith('Services:')) {
                // Additional info styling
                ctx.fillStyle = '#495057';
                ctx.font = '14px "Segoe UI", sans-serif';
                ctx.fillText(line, 70, yPosition + 15);
                yPosition += 20;
            }
            
            yPosition += lineHeight;
        });

        // Footer
        ctx.fillStyle = '#8c1965';
        ctx.fillRect(0, canvas.height - 40, canvas.width, 40);
        ctx.fillStyle = 'white';
        ctx.font = '12px "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Confidential - © 2025 Minute Long Solutions, LLC', canvas.width/2, canvas.height - 15);

        // Export with professional filename
        const link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = `Chat_Transcript_${now.toISOString().slice(0,10)}_${now.getHours()}${now.getMinutes()}.png`;
        link.click();
        
    } catch (error) {
        console.error('Failed to save screenshot:', error);
        alert('Error generating transcript. Please try again.');
    }
}

// Initialize help modal functionality
document.addEventListener('DOMContentLoaded', function() {
    const settingsForm = document.getElementById('settingsForm');
    
    // Initialize tooltips and popovers
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => new bootstrap.Tooltip(tooltip));
    
    const popovers = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popovers.map(popover => new bootstrap.Popover(popover));

    // Only initialize settings-related functionality if the form exists
    if (settingsForm) {
        // Load settings when page loads
        loadSettings();

        // Add smooth scrolling for help modal sections
        document.querySelectorAll('#helpModal .accordion-button').forEach(button => {
            button.addEventListener('click', function() {
                setTimeout(() => {
                    const content = this.nextElementSibling;
                    content.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 350);
            });
        });

        // Handle navigation active state
        const currentPath = window.location.pathname;
        document.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });

        // Track help modal usage
        const helpModal = document.getElementById('helpModal');
        if (helpModal) {
            helpModal.addEventListener('shown.bs.modal', function() {
                // You can add analytics tracking here if needed
                console.log('Help modal opened');
            });

            // Handle section visibility in URL
            const urlParams = new URLSearchParams(window.location.search);
            const section = urlParams.get('help_section');
            if (section) {
                const sectionButton = document.querySelector(`#heading${section} .accordion-button`);
                if (sectionButton && !sectionButton.classList.contains('collapsed')) {
                    sectionButton.click();
                }
            }
        }
    }
});
async function loadSettings() {
    // First check if we're on a page with settings form
    const settingsForm = document.getElementById('settingsForm');
    if (!settingsForm) {
        console.debug('Settings form not found, skipping settings load');
        return;
    }

    try {
        const response = await fetch('/api/settings');
        const settings = await response.json();

        // Helper function to safely set form values
        const setElementValue = (id, value) => {
            const element = document.getElementById(id);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                }
            }
        };

        // Helper function to safely set text content
        const setElementText = (id, value) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        };

        // Update all form fields safely
        setElementValue('sessionTimeout', settings.sessionTimeout);
        setElementValue('entityRecognition', settings.entityRecognition);
        setElementValue('sentimentAnalysis', settings.sentimentAnalysis);
        setElementValue('responseStyle', settings.responseStyle);
        setElementValue('confidenceThreshold', settings.confidenceThreshold);
        setElementValue('contextMemory', settings.contextMemory);
        setElementValue('autoSummarization', settings.autoSummarization);
        setElementValue('profanityFilter', settings.profanityFilter);
        setElementValue('memoryDuration', settings.memoryDuration);
        
        // Update threshold value display
        setElementText('thresholdValue', settings.confidenceThreshold);
        
        // Handle language support select
        const languageSelect = document.getElementById('languageSupport');
        if (languageSelect && settings.languageSupport) {
            Array.from(languageSelect.options).forEach(option => {
                option.selected = settings.languageSupport.includes(option.value);
            });
        }

    } catch (error) {
        console.error('Error loading settings:', error);
        showNotification('Error loading settings', 'error');
    }
}
