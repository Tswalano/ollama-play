class ChatApp {
    constructor() {
        this.currentConversationId = null;
        this.setupElements();
        this.setupEventListeners();
        this.initialize();
    }

    setupElements() {
        this.chatHistory = document.getElementById('chat-history');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatForm = document.getElementById('chat-form');
        this.userInput = document.getElementById('user-input');
        this.newChatButton = document.getElementById('new-chat');
        this.welcomeMessage = document.getElementById('welcome-message');
        this.startChatButton = document.getElementById('start-chat');
        this.thinkingIndicator = document.getElementById('thinking-indicator');
    }

    setupEventListeners() {
        this.newChatButton.addEventListener('click', () => this.startNewChat());
        this.startChatButton.addEventListener('click', () => this.startNewChat());
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        
        // Listen for URL changes
        window.addEventListener('popstate', (event) => {
            if (event.state && event.state.conversationId) {
                this.loadConversation(event.state.conversationId);
            }
        });
    }

    async initialize() {
        await this.loadConversations();
        
        // Check URL for conversation ID
        const urlParams = new URLSearchParams(window.location.search);
        const conversationId = urlParams.get('conversationId');
        if (conversationId) {
            this.currentConversationId = conversationId;
            await this.loadConversation(conversationId);
        }
    }

    async loadConversations() {
        try {
            const response = await fetch('/conversations');
            const data = await response.json();
            this.updateChatHistory(data.conversations);
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }

    updateChatHistory(conversations) {
        this.chatHistory.innerHTML = conversations.map(conv => `
            <div class="px-4 py-2 conversation-item ${conv.id === this.currentConversationId ? 'active' : ''}"
                 onclick="chatApp.loadConversation('${conv.id}')">
                <div class="text-sm font-medium truncate">${conv.title}</div>
                <div class="text-xs text-muted-foreground">
                    ${new Date(conv.start_time).toLocaleString()}
                </div>
            </div>
        `).join('');
    }

    toggleThinkingIndicator(show) {
        this.thinkingIndicator.style.display = show ? 'block' : 'none';
        const submitButton = this.chatForm.querySelector('button[type="submit"]');
        submitButton.disabled = show;
    }

    async startNewChat() {
        this.welcomeMessage.style.display = 'none';
        
        const formattedDate = new Date().toLocaleString('en-GB', {
            weekday: 'short',
            day: '2-digit',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit'
        });

        try {
            const response = await fetch('/conversation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: 'New Chat: ' + formattedDate })
            });
            
            const data = await response.json();
            this.currentConversationId = data.id;

            // Update URL
            window.history.pushState(
                { conversationId: this.currentConversationId },
                '',
                `?conversationId=${this.currentConversationId}`
            );

            this.chatMessages.innerHTML = '';
            await this.loadConversations();
        } catch (error) {
            console.error('Error creating new chat:', error);
        }
    }

    async loadConversation(conversationId) {
        try {
            const response = await fetch(`/conversation/${conversationId}`);
            if (!response.ok) {
                throw new Error('Failed to fetch conversation');
            }

            const data = await response.json();
            this.currentConversationId = conversationId;

            // Update URL if it's different
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('conversationId') !== conversationId) {
                window.history.pushState(
                    { conversationId },
                    '',
                    `?conversationId=${conversationId}`
                );
            }

            this.welcomeMessage.style.display = 'none';
            this.chatMessages.innerHTML = '';
            
            // Render messages
            data.conversation.forEach(msg => {
                this.appendMessage(msg);
            });

            // Update active state in sidebar
            document.querySelectorAll('.conversation-item').forEach(item => {
                item.classList.toggle('active', item.getAttribute('data-id') === conversationId);
            });

            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        } catch (error) {
            console.error('Error loading conversation:', error);
        }
    }

    appendMessage(message) {
        const isUser = message.role === 'user';
        const messageHTML = `
            <div class="chat-message ${isUser ? 'user' : 'assistant'}">
                <div class="message-content">
                    ${message.content}
                </div>
                <div class="text-xs text-muted-foreground mt-1">
                    ${new Date(message.timestamp).toLocaleString()}
                </div>
            </div>
        `;
        this.chatMessages.insertAdjacentHTML('beforeend', messageHTML);
    }

    async handleSubmit(event) {
        event.preventDefault();
        const message = this.userInput.value.trim();
        if (!message) return;

        if (!this.currentConversationId) {
            await this.startNewChat();
        }

        this.userInput.value = '';
        this.appendMessage({
            role: 'user',
            content: message,
            timestamp: new Date()
        });

        this.toggleThinkingIndicator(true);

        try {
            const response = await fetch(`/conversation/${this.currentConversationId}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: message,
                    conversation_id: this.currentConversationId
                })
            });

            const data = await response.json();
            this.toggleThinkingIndicator(false);

            this.appendMessage({
                role: 'assistant',
                content: data.html,
                timestamp: new Date()
            });

            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        } catch (error) {
            console.error('Error sending message:', error);
            this.toggleThinkingIndicator(false);
        }
    }
}

// Initialize the chat application
const chatApp = new ChatApp();

// Make it globally available for onclick handlers
window.chatApp = chatApp;