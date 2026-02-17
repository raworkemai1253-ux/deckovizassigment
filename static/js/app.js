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
let currentMode = 'auto'; // 'auto', 'image', 'video'
let recognition = null;

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
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const previewClose = document.getElementById('previewClose');

// New Elements
const modeBtns = document.querySelectorAll('.mode-btn');
const micBtn = document.getElementById('micBtn');


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

// ─── Mode Selection ─────────────────────────────────────────────────────
modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all
        modeBtns.forEach(b => b.classList.remove('active'));
        // Add to clicked
        btn.classList.add('active');
        // Set mode
        currentMode = btn.getAttribute('data-mode');
        console.log('Mode switched to:', currentMode);
    });
});


// ─── Speech to Text ─────────────────────────────────────────────────────
function setupSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            micBtn.classList.add('listening');
        };

        recognition.onend = () => {
            micBtn.classList.remove('listening');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            messageInput.value += (messageInput.value ? ' ' : '') + transcript;
            // Trigger input event to resize/enable send
            messageInput.dispatchEvent(new Event('input'));
        };

        micBtn.addEventListener('click', () => {
            if (micBtn.classList.contains('listening')) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    } else {
        if (micBtn) micBtn.style.display = 'none'; // Hide if not supported
    }
}


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
            // Only use <video> for actual video files, not GIFs
            const videoExtRegex = /\.(mp4|webm|mov)(\?|$)/i;
            const isRealVideo = videoExtRegex.test(gc.image_url || '');
            const mediaHtml = isRealVideo
                ? `<video src="${gc.image_url}" autoplay loop muted playsinline controls style="width:100%;aspect-ratio:16/9;object-fit:cover;display:block;border-radius:12px 12px 0 0;"></video>`
                : `<img src="${gc.image_url}" alt="${escapeHtml(gc.title)}" loading="lazy">`;

            gridHtml += `
                <div class="content-card" onclick="selectImage('${gc.image_url}', this)">
                    ${mediaHtml}
                    <div class="content-card-overlay">
                        <div class="content-card-title">${escapeHtml(gc.title)}</div>
                        <div class="content-card-actions">
                            ${!isRealVideo ? `<button class="btn-icon-small" onclick="event.stopPropagation(); openLightbox('${gc.image_url}', '${escapeHtml(gc.title)}')" title="View full size">
                                🔍
                            </button>` : ''}
                            <button class="btn-icon-small" onclick="event.stopPropagation(); downloadImage('${gc.image_url}', '${escapeHtml(gc.title)}')" title="Download">
                                ⬇️
                            </button>
                        </div>
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


// ─── Image Upload ────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'flex';
            sendBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

previewClose.addEventListener('click', () => {
    imageInput.value = '';
    imagePreview.style.display = 'none';
    previewImg.src = '';
    if (messageInput.value.trim().length === 0) {
        sendBtn.disabled = true;
    }
});

// ─── Selection / Refinement ──────────────────────────────────────────────
let selectedImageUrl = null;

function selectImage(url, cardElement) {
    if (selectedImageUrl === url) {
        // Deselect
        selectedImageUrl = null;
        cardElement.classList.remove('selected');
        // Remove input hint
        messageInput.placeholder = "Describe what you'd like to create...";
    } else {
        // Deselect others
        document.querySelectorAll('.content-card').forEach(c => c.classList.remove('selected'));
        // Select this
        selectedImageUrl = url;
        cardElement.classList.add('selected');
        // Visual feedback
        messageInput.placeholder = "Refine this image (e.g., 'make it darker')...";
        messageInput.focus();
    }
}


// ─── Sending Messages ────────────────────────────────────────────────────
async function sendMessage() {
    const content = messageInput.value.trim();
    const imageFile = imageInput.files[0];

    if ((!content && !imageFile) || isLoading) return;

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

    // Clear flags but keep input/selection logic until sent
    // Actually standard chat clears input immediately.

    // UI Updates
    messageInput.value = '';
    messageInput.style.height = 'auto';
    messageInput.placeholder = "Describe what you'd like to create...";

    if (imageFile) {
        imageInput.value = '';
        imagePreview.style.display = 'none';
        previewImg.src = '';
    }

    // Clear selection visual but keep URL for sending
    const currentRefinementUrl = selectedImageUrl;
    selectedImageUrl = null;
    document.querySelectorAll('.content-card').forEach(c => c.classList.remove('selected'));

    sendBtn.disabled = true;
    isLoading = true;

    // Show user message immediately
    const tempUserMsg = {
        id: 'temp-user',
        role: 'user',
        content: content || (imageFile ? '[Image Uploaded]' : ''),
        generated_contents: [],
    };
    appendMessage(tempUserMsg);
    scrollToBottom();

    // Show typing indicator
    typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
        let data;
        // Construct payload including MODE & Refinement
        if (imageFile) {
            const formData = new FormData();
            formData.append('conversation_id', currentConversationId);
            formData.append('content', content);
            formData.append('image', imageFile);
            formData.append('mode', currentMode);
            if (currentRefinementUrl) formData.append('refinement_url', currentRefinementUrl);

            const response = await fetch('/api/messages/', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `API error ${response.status}`);
            }
            data = await response.json();
        } else {
            console.log("Sending mode:", currentMode);
            data = await api('/api/messages/', {
                method: 'POST',
                body: JSON.stringify({
                    conversation_id: currentConversationId,
                    content: content,
                    mode: currentMode,
                    refinement_url: currentRefinementUrl, // Send selection
                }),
            });
        }

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
setupSpeechRecognition();

// ─── Download Image ──────────────────────────────────────────────────────
async function downloadImage(url, title) {
    try {
        const response = await fetch(url);
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        const ext = url.split('.').pop().split('?')[0] || 'png';
        const safeName = (title || 'vizzy-image').replace(/[^a-z0-9]/gi, '_').substring(0, 40);
        a.download = `${safeName}.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(blobUrl);
    } catch (err) {
        // Fallback: open in new tab
        window.open(url, '_blank');
    }
}
