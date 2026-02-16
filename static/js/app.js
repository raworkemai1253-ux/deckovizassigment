/**
 * Vizzy Chat — Frontend Application
 *
 * Handles all client-side logic:
 * - Conversation CRUD (create, list, switch, rename, delete)
 * - Message sending and receiving
 * - Dynamic rendering of text + image content
 * - Typing indicators, auto-scroll, keyboard shortcuts
 * - Regeneration of AI responses
 */

// ─── State ────────────────────────────────────────────────────────────────
let currentConversationId = null;
let conversations = [];
let isLoading = false;

// ─── DOM Elements ─────────────────────────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarClose = document.getElementById('sidebarClose');
const newChatBtn = document.getElementById('newChatBtn');
const conversationsList = document.getElementById('conversationsList');
const chatTitle = document.getElementById('chatTitle');
const clearChatBtn = document.getElementById('clearChatBtn');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesEl = document.getElementById('messages');
const typingIndicator = document.getElementById('typingIndicator');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');

// ─── API Helpers ──────────────────────────────────────────────────────────
async function api(url, options = {}) {
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const response = await fetch(url, { ...defaults, ...options });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `API error ${response.status}`);
    }
    return response.json();
}

// ─── Sidebar Toggle (Mobile) ─────────────────────────────────────────────
function openSidebar() {
    sidebar.classList.add('open');
    sidebarOverlay.classList.add('visible');
}

function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('visible');
}

sidebarToggle.addEventListener('click', openSidebar);
sidebarClose.addEventListener('click', closeSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);

// ─── Conversations ────────────────────────────────────────────────────────
async function loadConversations() {
    try {
        const data = await api('/api/conversations/');
        conversations = data.conversations;
        renderConversations();
    } catch (err) {
        console.error('Failed to load conversations:', err);
    }
}

function renderConversations() {
    conversationsList.innerHTML = '';

    if (conversations.length === 0) {
        conversationsList.innerHTML = `
            <div style="padding: 20px 12px; text-align: center; color: var(--text-muted); font-size: 12px;">
                No conversations yet.<br>Start a new chat!
            </div>
        `;
        return;
    }

    conversations.forEach(conv => {
        const div = document.createElement('div');
        div.className = `conversation-item${conv.id === currentConversationId ? ' active' : ''}`;
        div.setAttribute('data-id', conv.id);
        div.innerHTML = `
            <div class="conversation-item-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
                </svg>
            </div>
            <span class="conversation-item-title">${escapeHtml(conv.title)}</span>
            <button class="conversation-item-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')" title="Delete">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
            </button>
        `;
        div.addEventListener('click', () => switchConversation(conv.id));
        conversationsList.appendChild(div);
    });
}

async function createConversation() {
    try {
        const data = await api('/api/conversations/', {
            method: 'POST',
            body: JSON.stringify({ title: 'New Chat' }),
        });
        currentConversationId = data.id;
        await loadConversations();
        showChat();
        messagesEl.innerHTML = '';
        chatTitle.textContent = data.title;
        clearChatBtn.style.display = 'flex';
        messageInput.focus();
        closeSidebar();
    } catch (err) {
        console.error('Failed to create conversation:', err);
    }
}

async function switchConversation(id) {
    if (id === currentConversationId) return;
    currentConversationId = id;
    renderConversations();
    closeSidebar();

    try {
        const data = await api(`/api/conversations/${id}/`);
        chatTitle.textContent = data.title;
        clearChatBtn.style.display = 'flex';
        showChat();
        renderMessages(data.messages);
        scrollToBottom();
    } catch (err) {
        console.error('Failed to load conversation:', err);
    }
}

async function deleteConversation(id) {
    try {
        await api(`/api/conversations/${id}/`, { method: 'DELETE' });
        if (currentConversationId === id) {
            currentConversationId = null;
            showWelcome();
            chatTitle.textContent = 'Vizzy Chat';
            clearChatBtn.style.display = 'none';
        }
        await loadConversations();
    } catch (err) {
        console.error('Failed to delete conversation:', err);
    }
}

newChatBtn.addEventListener('click', createConversation);
clearChatBtn.addEventListener('click', () => {
    if (currentConversationId) deleteConversation(currentConversationId);
});

// ─── Views ────────────────────────────────────────────────────────────────
function showWelcome() {
    welcomeScreen.style.display = 'flex';
    messagesEl.style.display = 'none';
}

function showChat() {
    welcomeScreen.style.display = 'none';
    messagesEl.style.display = 'block';
}

// ─── Message Rendering ───────────────────────────────────────────────────
function renderMessages(messages) {
    messagesEl.innerHTML = '';
    messages.forEach(msg => {
        appendMessage(msg, false);
    });
}

function appendMessage(msg, animate = true) {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;
    div.setAttribute('data-id', msg.id);
    if (!animate) div.style.animation = 'none';

    const avatarContent = msg.role === 'user'
        ? 'U'
        : `<svg width="18" height="18" viewBox="0 0 28 28" fill="none">
             <path d="M9 14L12 17L19 10" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
           </svg>`;

    let contentHtml = `<div class="message-content">${formatText(msg.content)}</div>`;

    // Render generated content grid
    if (msg.generated_contents && msg.generated_contents.length > 0) {
        const count = msg.generated_contents.length;
        const gridClass = `grid-${Math.min(count, 6)}`;
        let gridHtml = `<div class="content-grid ${gridClass}">`;

        msg.generated_contents.forEach(gc => {
            gridHtml += `
                <div class="content-card" onclick="openLightbox('${gc.image_url}', '${escapeHtml(gc.title)}')">
                    <img src="${gc.image_url}" alt="${escapeHtml(gc.title)}" loading="lazy">
                    <div class="content-card-overlay">
                        <div class="content-card-title">${escapeHtml(gc.title)}</div>
                        <div class="content-card-type">${gc.content_type}</div>
                    </div>
                </div>
            `;
        });

        gridHtml += '</div>';
        contentHtml += gridHtml;
    }

    // Add action buttons for assistant messages
    if (msg.role === 'assistant') {
        contentHtml += `
            <div class="message-actions">
                <button class="action-btn" onclick="regenerateResponse('${msg.id}')">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M1 4v6h6M23 20v-6h-6"/>
                        <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/>
                    </svg>
                    Regenerate
                </button>
                <button class="action-btn" onclick="copyText(this, \`${escapeForTemplate(msg.content)}\`)">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                    </svg>
                    Copy
                </button>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="message-avatar">${avatarContent}</div>
        <div class="message-body">${contentHtml}</div>
    `;

    messagesEl.appendChild(div);
}

function formatText(text) {
    // Basic markdown-like formatting
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeForTemplate(str) {
    return str.replace(/`/g, '\\`').replace(/\$/g, '\\$').replace(/\\/g, '\\\\');
}

// ─── Sending Messages ────────────────────────────────────────────────────
async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || isLoading) return;

    // Create conversation if needed
    if (!currentConversationId) {
        try {
            const data = await api('/api/conversations/', {
                method: 'POST',
                body: JSON.stringify({ title: 'New Chat' }),
            });
            currentConversationId = data.id;
            showChat();
            clearChatBtn.style.display = 'flex';
        } catch (err) {
            console.error('Failed to create conversation:', err);
            return;
        }
    }

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;
    isLoading = true;

    // Show user message immediately
    const tempUserMsg = {
        id: 'temp-user',
        role: 'user',
        content: content,
        generated_contents: [],
    };
    appendMessage(tempUserMsg);
    scrollToBottom();

    // Show typing indicator
    typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
        const data = await api('/api/messages/', {
            method: 'POST',
            body: JSON.stringify({
                conversation_id: currentConversationId,
                content: content,
            }),
        });

        // Remove temp user message and add real ones
        const tempEl = messagesEl.querySelector('[data-id="temp-user"]');
        if (tempEl) tempEl.remove();

        // Hide typing indicator
        typingIndicator.style.display = 'none';

        appendMessage(data.user_message, false);
        appendMessage(data.assistant_message);

        // Update conversation title
        if (data.conversation_title) {
            chatTitle.textContent = data.conversation_title;
        }

        scrollToBottom();
        await loadConversations();
    } catch (err) {
        console.error('Failed to send message:', err);
        typingIndicator.style.display = 'none';
        // Show error
        const errorMsg = {
            id: 'error-' + Date.now(),
            role: 'assistant',
            content: 'Sorry, something went wrong. Please try again.',
            generated_contents: [],
        };
        appendMessage(errorMsg);
        scrollToBottom();
    } finally {
        isLoading = false;
    }
}

// ─── Regeneration ────────────────────────────────────────────────────────
async function regenerateResponse(messageId) {
    if (isLoading) return;
    isLoading = true;

    // Find and replace the existing message
    const msgEl = messagesEl.querySelector(`[data-id="${messageId}"]`);
    if (msgEl) {
        const body = msgEl.querySelector('.message-body');
        body.innerHTML = `
            <div class="message-content" style="color: var(--text-muted);">
                <em>Regenerating response...</em>
            </div>
        `;
    }

    try {
        const data = await api(`/api/messages/${messageId}/regenerate/`, {
            method: 'POST',
        });

        // Re-render the message
        if (msgEl) {
            msgEl.remove();
        }
        appendMessage(data);
        scrollToBottom();
    } catch (err) {
        console.error('Failed to regenerate:', err);
        if (msgEl) {
            const body = msgEl.querySelector('.message-body');
            body.innerHTML = `
                <div class="message-content" style="color: var(--danger);">
                    Failed to regenerate. Please try again.
                </div>
            `;
        }
    } finally {
        isLoading = false;
    }
}

// ─── Copy Text ───────────────────────────────────────────────────────────
function copyText(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = btn.innerHTML;
        btn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 6L9 17l-5-5"/>
            </svg>
            Copied!
        `;
        setTimeout(() => { btn.innerHTML = originalText; }, 1500);
    });
}

// ─── Lightbox ────────────────────────────────────────────────────────────
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightboxImage');
const lightboxCaption = document.getElementById('lightboxCaption');

function openLightbox(url, caption) {
    lightboxImage.src = url;
    lightboxCaption.textContent = caption;
    lightbox.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    lightbox.style.display = 'none';
    document.body.style.overflow = '';
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.style.display === 'flex') {
        closeLightbox();
    }
});

// ─── Suggestion Cards ────────────────────────────────────────────────────
function useSuggestion(btn) {
    const text = btn.querySelector('span').textContent;
    messageInput.value = text;
    sendBtn.disabled = false;
    sendMessage();
}

// ─── Auto-resize Textarea ────────────────────────────────────────────────
messageInput.addEventListener('input', () => {
    sendBtn.disabled = messageInput.value.trim().length === 0;
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
});

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

// ─── Scroll ──────────────────────────────────────────────────────────────
function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
}

// ─── Init ────────────────────────────────────────────────────────────────
loadConversations();
