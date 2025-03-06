document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    const ticketModal = new bootstrap.Modal(document.getElementById('ticketModal'), {
        backdrop: false,
        keyboard: true,
        focus: true
    });
    
    const deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'), {
        backdrop: false,
        keyboard: true
    });
    
    const helpModal = new bootstrap.Modal(document.getElementById('helpModal'), {
        keyboard: true
    });
    
    const toast = new bootstrap.Toast(document.getElementById('notificationToast'));
    
    // Get CSRF token
    const csrfToken = document.querySelector('input[name="csrf_token"]').value;
    
    // Cache DOM elements
    const ticketsList = document.getElementById('ticketsList');
    const ticketForm = document.getElementById('ticketForm');
    const searchInput = document.getElementById('searchTickets');
    const statusFilter = document.getElementById('statusFilter');
    const priorityFilter = document.getElementById('priorityFilter');
    const loadingSpinner = document.getElementById('loading-spinner');
    const modalElement = document.getElementById('ticketModal');
    const helpModalElement = document.getElementById('helpModal');
    
    // Handle modal cleanup
    modalElement.addEventListener('hidden.bs.modal', function () {
        ticketForm.reset();
        ticketForm.classList.remove('was-validated');
        document.body.classList.remove('modal-open');
    });
    
    // Current sort state
    let currentSort = {
        column: 'created_at',
        direction: 'desc'
    };
    
    // Initialize WebSocket connection
    const socket = io();
    
    socket.on('connect', () => {
        console.log('WebSocket connected');
        showNotification('Real-time updates enabled', 'info');
        // Join the tickets room for updates
        socket.emit('join', { room: 'tickets' });
    });

    socket.on('connect_error', () => {
        console.error('WebSocket connection failed');
        showNotification('Real-time updates unavailable', 'warning');
    });
    
    // Enhanced WebSocket event handlers for real-time updates
    socket.on('ticket_update', (data) => {
        console.log('Ticket update received:', data);
        loadTickets();
        showNotification('Ticket updated in real-time', 'info');
    });

    socket.on('ticket_create', (data) => {
        console.log('New ticket created:', data);
        loadTickets();
        showNotification('New ticket created', 'success');
    });

    socket.on('ticket_delete', (data) => {
        console.log('Ticket deleted:', data);
        loadTickets();
        showNotification('Ticket deleted', 'info');
    });
    
    // Load tickets on page load
    loadTickets();
    
    // Event Listeners
    document.getElementById('createTicketBtn').addEventListener('click', () => {
        document.getElementById('modalTitle').textContent = 'Create New Ticket';
        ticketForm.reset();
        ticketForm.dataset.mode = 'create';
        delete ticketForm.dataset.ticketId;
        ticketForm.classList.remove('was-validated');
        document.body.classList.add('modal-open');
    });
    
    // Add debounce function at the top level
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Cache the refresh button
    const refreshButton = document.getElementById('refreshTickets');
    let isRefreshing = false;

    // Debounced refresh function
    const debouncedRefresh = debounce(async () => {
        if (isRefreshing) return;
        
        isRefreshing = true;
        refreshButton.disabled = true;
        refreshButton.innerHTML = '<i class="fas fa-sync fa-spin"></i> Refreshing...';
        
        try {
            await loadTickets();
            showNotification('Tickets refreshed successfully', 'success');
        } catch (error) {
            console.error('Refresh error:', error);
            showNotification('Failed to refresh tickets. Please try again.', 'danger');
        } finally {
            isRefreshing = false;
            refreshButton.disabled = false;
            refreshButton.innerHTML = '<i class="fas fa-sync"></i> Refresh';
        }
    }, 300);

    // Update the refresh button event listener
    refreshButton.addEventListener('click', (e) => {
        e.preventDefault();
        debouncedRefresh();
    });
    
    // Sort tickets when clicking on table headers
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.sort;
            // Update sort icons
            document.querySelectorAll('th.sortable i').forEach(icon => {
                icon.className = 'fas fa-sort';
            });
            
            if (currentSort.column === column) {
                currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
            } else {
                currentSort.column = column;
                currentSort.direction = 'asc';
            }
            
            // Update icon for current sort column
            const icon = th.querySelector('i');
            icon.className = `fas fa-sort-${currentSort.direction === 'asc' ? 'up' : 'down'}`;
            
            loadTickets();
        });
    });
    
    // Filter and search handlers with debounce
    let filterTimeout;
    const handleFilterChange = () => {
        clearTimeout(filterTimeout);
        filterTimeout = setTimeout(() => {
            loadTickets();
        }, 300);
    };
    
    [statusFilter, priorityFilter].forEach(filter => {
        filter.addEventListener('change', handleFilterChange);
    });
    
    // Add debounced search function
    function searchTickets(searchTerm) {
        const searchTermLower = searchTerm.toLowerCase();
        const rows = document.querySelectorAll('#ticketsTable tbody tr:not(.no-tickets-message)');
        let visibleCount = 0;
        
        rows.forEach(row => {
            const description = row.querySelector('td:nth-child(2)').textContent.toLowerCase();
            const priority = row.querySelector('td:nth-child(3)').textContent.toLowerCase();
            const status = row.querySelector('td:nth-child(4)').textContent.toLowerCase();
            const created = row.querySelector('td:nth-child(5)').textContent.toLowerCase();
            
            if (description.includes(searchTermLower) || 
                priority.includes(searchTermLower) || 
                status.includes(searchTermLower) || 
                created.includes(searchTermLower)) {
                row.style.display = '';
                visibleCount++;
            } else {
                row.style.display = 'none';
            }
        });
        
        // Show/hide no results message
        const noTicketsMessage = document.querySelector('.no-tickets-message');
        if (noTicketsMessage) {
            if (visibleCount === 0) {
                noTicketsMessage.style.display = '';
            } else {
                noTicketsMessage.style.display = 'none';
            }
        }
    }

    // Initialize search with debounce
    if (searchInput) {
        const debouncedSearch = debounce((value) => {
            searchTickets(value);
        }, 300);
        
        searchInput.addEventListener('input', (e) => {
            debouncedSearch(e.target.value);
        });
    }
    
    // Modify the form submission handler to emit socket events
    ticketForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (!ticketForm.checkValidity()) {
            ticketForm.classList.add('was-validated');
            return;
        }
        
        const submitButton = ticketForm.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
        
        const formData = {
            description: document.getElementById('description').value.trim(),
            priority: document.getElementById('priority').value,
            status: document.getElementById('status').value
        };
        
        const mode = ticketForm.dataset.mode;
        const ticketId = ticketForm.dataset.ticketId;
        
        try {
            const response = await fetch(`/api/tickets${mode === 'edit' ? `/${ticketId}` : ''}`, {
                method: mode === 'create' ? 'POST' : 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to save ticket');
            }
            
            const result = await response.json();
            
            // Emit socket event for real-time updates
            socket.emit(mode === 'create' ? 'ticket_create' : 'ticket_update', {
                ticket: result,
                action: mode
            });
            
            showNotification(`Ticket ${mode === 'create' ? 'created' : 'updated'} successfully`, 'success');
            ticketModal.hide();
            
        } catch (error) {
            console.error('Error:', error);
            showNotification(error.message || 'Failed to save ticket', 'danger');
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = originalText;
        }
    });
    
    // Modify the delete confirmation handler to emit socket events
    let ticketToDelete = null;
    
    document.getElementById('confirmDelete').addEventListener('click', async () => {
        if (!ticketToDelete) return;
        
        const deleteButton = document.getElementById('confirmDelete');
        const originalText = deleteButton.innerHTML;
        deleteButton.disabled = true;
        deleteButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Deleting...';
        
        try {
            const response = await fetch(`/api/tickets/${ticketToDelete}`, {
                method: 'DELETE',
                headers: {
                    'X-CSRFToken': csrfToken
                }
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to delete ticket');
            }
            
            // Emit socket event for real-time updates
            socket.emit('ticket_delete', {
                ticketId: ticketToDelete
            });
            
            showNotification('Ticket deleted successfully', 'success');
            deleteModal.hide();
            
        } catch (error) {
            console.error('Error:', error);
            showNotification(error.message || 'Failed to delete ticket', 'danger');
        } finally {
            deleteButton.disabled = false;
            deleteButton.innerHTML = originalText;
            ticketToDelete = null;
        }
    });
    
    // Helper Functions
    async function loadTickets() {
        showLoading(true);
        
        try {
            const searchQuery = searchInput.value.toLowerCase();
            const statusValue = statusFilter.value;
            const priorityValue = priorityFilter.value;
            
            const response = await fetch('/api/tickets');
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to fetch tickets');
            }
            
            let tickets = await response.json();
            
            // Apply filters
            tickets = tickets.filter(ticket => {
                const matchesSearch = ticket.description.toLowerCase().includes(searchQuery);
                const matchesStatus = !statusValue || ticket.status === statusValue;
                const matchesPriority = !priorityValue || ticket.priority === priorityValue;
                return matchesSearch && matchesStatus && matchesPriority;
            });
            
            // Apply sorting
            tickets.sort((a, b) => {
                let valueA = a[currentSort.column];
                let valueB = b[currentSort.column];
                
                if (currentSort.column === 'created_at') {
                    valueA = new Date(valueA);
                    valueB = new Date(valueB);
                }
                
                if (valueA < valueB) return currentSort.direction === 'asc' ? -1 : 1;
                if (valueA > valueB) return currentSort.direction === 'asc' ? 1 : -1;
                return 0;
            });
            
            renderTickets(tickets);
            
        } catch (error) {
            console.error('Error:', error);
            showNotification(error.message || 'Failed to load tickets', 'danger');
        } finally {
            showLoading(false);
        }
    }
    
    function renderTickets(tickets) {
        const noTicketsMessage = document.querySelector('.no-tickets-message');
        
        if (tickets.length === 0) {
            noTicketsMessage.style.display = 'table-row';
            ticketsList.innerHTML = '';
            return;
        }
        
        noTicketsMessage.style.display = 'none';
        ticketsList.innerHTML = tickets.map(ticket => `
            <tr>
                <td>${ticket.id}</td>
                <td>${escapeHtml(ticket.description)}</td>
                <td><span class="priority-${ticket.priority}">${capitalizeFirst(ticket.priority)}</span></td>
                <td><span class="status-badge status-${ticket.status}">${formatStatus(ticket.status)}</span></td>
                <td>${formatDate(ticket.created_at)}</td>
                <td>
                    <div class="action-buttons">
                        <button class="btn btn-sm btn-primary" onclick="editTicket(${ticket.id})" title="Edit ticket">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="deleteTicket(${ticket.id})" title="Delete ticket">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }
    
    window.editTicket = async function(id) {
        try {
            showLoading(true);
            const response = await fetch(`/api/tickets/${id}`);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Failed to fetch ticket');
            }
            
            const ticket = await response.json();
            
            document.getElementById('modalTitle').textContent = `Edit Ticket #${ticket.id}`;
            document.getElementById('ticketId').value = ticket.id;
            document.getElementById('description').value = ticket.description;
            document.getElementById('priority').value = ticket.priority;
            document.getElementById('status').value = ticket.status;
            
            ticketForm.dataset.mode = 'edit';
            ticketForm.dataset.ticketId = ticket.id;
            ticketForm.classList.remove('was-validated');
            ticketModal.show();
            
        } catch (error) {
            console.error('Error:', error);
            showNotification(error.message || 'Failed to load ticket details', 'danger');
        } finally {
            showLoading(false);
        }
    };
    
    window.deleteTicket = function(id) {
        ticketToDelete = id;
        document.querySelector('#deleteModal .modal-body p:first-child').textContent = `Are you sure you want to delete ticket #${id}?`;
        deleteModal.show();
    };
    
    function showNotification(message, type) {
        const toastBody = document.querySelector('.toast-body');
        const toastElement = document.getElementById('notificationToast');
        
        toastElement.className = `toast border-${type}`;
        toastBody.textContent = message;
        toast.show();
    }
    
    function showLoading(show) {
        loadingSpinner.style.display = show ? 'block' : 'none';
        document.querySelector('.table-responsive').style.opacity = show ? '0.5' : '1';
    }
    
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function capitalizeFirst(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    function formatStatus(status) {
        return status.split('_').map(capitalizeFirst).join(' ');
    }
    
    function formatDate(dateString) {
        if (!dateString) return 'N/A';
        try {
            return new Date(dateString).toLocaleString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            console.error('Error formatting date:', error);
            return dateString;
        }
    }

    // Handle help modal
    window.showHelpModal = function() {
        helpModal.show();
    };

    helpModalElement.addEventListener('hidden.bs.modal', function () {
        document.body.classList.remove('modal-open');
    });

    // Add auto-refresh interval (every 30 seconds)
    const AUTO_REFRESH_INTERVAL = 30000; // 30 seconds
    let autoRefreshInterval;

    function startAutoRefresh() {
        if (autoRefreshInterval) clearInterval(autoRefreshInterval);
        autoRefreshInterval = setInterval(() => {
            if (!document.hidden) { // Only refresh if page is visible
                loadTickets();
            }
        }, AUTO_REFRESH_INTERVAL);
    }

    // Start auto-refresh when page loads
    startAutoRefresh();

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            loadTickets(); // Refresh immediately when page becomes visible
        }
    });

    // Cleanup interval when page is unloaded
    window.addEventListener('beforeunload', () => {
        if (autoRefreshInterval) clearInterval(autoRefreshInterval);
    });
}); 