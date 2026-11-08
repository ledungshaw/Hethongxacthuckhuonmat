/**
 * ============================================
 *  HỆ THỐNG CHECK-IN SÂN BAY
 *  Main JavaScript
 * ============================================
 */

// ========== KHỞI TẠO HỆ THỐNG ==========
console.log('✈️ Hệ thống Check-in Sân bay đã sẵn sàng!');
console.log('📋 Chào mừng bạn đến với hệ thống!');

// ========== UTILITY FUNCTIONS ==========

/**
 * Format ngày giờ theo định dạng Việt Nam
 */
function formatDateTime(dateStr) {
    if (!dateStr) return '--';
    try {
        return new Date(dateStr).toLocaleString('vi-VN');
    } catch (e) {
        return dateStr;
    }
}

/**
 * Format ngày theo định dạng DD/MM/YYYY
 * Hỗ trợ: "2024-01-15", "2024-01-15T00:00:00", Date object, null, undefined
 */
function formatDate(dateStr) {
    if (!dateStr || dateStr === 'null' || dateStr === 'undefined') return '--';
    try {
        // Trường hợp là ISO string YYYY-MM-DD hoặc YYYY-MM-DDTHH:mm:ss
        if (typeof dateStr === 'string') {
            // Bỏ phần time nếu có
            var cleanStr = dateStr.split('T')[0];
            var parts = cleanStr.split('-');
            if (parts.length === 3 && parts[0].length === 4) {
                return parts[2] + '/' + parts[1] + '/' + parts[0];
            }
        }
        // Fallback: dùng Date object
        var d = new Date(dateStr);
        if (isNaN(d.getTime())) return '--';
        return d.getDate().toString().padStart(2, '0') + '/' +
               (d.getMonth() + 1).toString().padStart(2, '0') + '/' +
               d.getFullYear();
    } catch (e) {
        return '--';
    }
}

/**
 * Hiển thị thông báo alert đẹp hơn
 */
function showAlert(message, type) {
    type = type || 'info';
    var alertBox = document.createElement('div');
    alertBox.className = 'alert ' + type;
    alertBox.textContent = message;
    alertBox.style.position = 'fixed';
    alertBox.style.top = '20px';
    alertBox.style.right = '20px';
    alertBox.style.zIndex = '9999';
    alertBox.style.minWidth = '250px';
    alertBox.style.boxShadow = '0 5px 20px rgba(0,0,0,0.15)';
    alertBox.style.animation = 'slideIn 0.3s ease';
    
    document.body.appendChild(alertBox);
    
    setTimeout(function() {
        alertBox.style.animation = 'slideOut 0.3s ease';
        setTimeout(function() {
            alertBox.remove();
        }, 300);
    }, 3000);
}

/**
 * Debounce function để tránh gọi API quá nhiều lần
 */
function debounce(func, wait) {
    var timeout;
    return function executedFunction() {
        var context = this;
        var args = arguments;
        var later = function() {
            timeout = null;
            func.apply(context, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ========== ANIMATION STYLES ==========
(function addAnimationStyles() {
    var style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
})();

// ========== AUTO REFRESH ==========
/**
 * Tự động refresh dữ liệu mỗi 30 giây
 */
function startAutoRefresh(callback, interval) {
    interval = interval || 30000;
    setInterval(callback, interval);
}

// ========== EXPORT CHO CÁC TRANG KHÁC ==========
window.AirportSystem = {
    formatDateTime: formatDateTime,
    formatDate: formatDate,
    showAlert: showAlert,
    debounce: debounce,
    startAutoRefresh: startAutoRefresh
};