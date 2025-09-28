// Academic Homepage JavaScript

// Update last modified date
document.addEventListener('DOMContentLoaded', function() {
    const lastUpdatedElement = document.getElementById('last-updated');
    if (lastUpdatedElement) {
        const currentDate = new Date();
        const options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        lastUpdatedElement.textContent = currentDate.toLocaleDateString('en-US', options);
    }
    
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.navigation a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Highlight active navigation link based on scroll position
    function updateActiveNavLink() {
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.navigation a');
        
        let currentSection = '';
        
        sections.forEach(section => {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100) {
                currentSection = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + currentSection) {
                link.classList.add('active');
            }
        });
    }
    
    // Update active nav link on scroll
    window.addEventListener('scroll', updateActiveNavLink);
    
    // Check for missing files and show helpful messages
    const resumeLinks = document.querySelectorAll('.download-link');
    resumeLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href.includes('files/') && (href.includes('resume.pdf') || href.includes('cv.pdf'))) {
                // This is a placeholder link - could add more sophisticated checking
                console.log('Resume/CV link clicked:', href);
            }
        });
    });
});

// Function to create a new news item (for easy content management)
function addNewsItem(date, content) {
    const newsContent = document.querySelector('.news-content');
    const noteElement = newsContent.querySelector('.note');
    
    const newsItem = document.createElement('div');
    newsItem.className = 'news-item';
    newsItem.innerHTML = `
        <span class="date">${date}</span>
        <p>${content}</p>
    `;
    
    // Insert before the note
    newsContent.insertBefore(newsItem, noteElement);
}

// Function to add a new publication (for easy content management)
function addPublication(title, authors, venue, links = {}) {
    const publicationsContent = document.querySelector('.publications-content');
    const noteElement = publicationsContent.querySelector('.note');
    
    const publicationItem = document.createElement('div');
    publicationItem.className = 'publication-item';
    
    let linksHtml = '';
    for (const [linkText, linkUrl] of Object.entries(links)) {
        linksHtml += `<a href="${linkUrl}" class="pub-link">${linkText}</a>`;
    }
    
    publicationItem.innerHTML = `
        <h3>${title}</h3>
        <p class="authors">${authors}</p>
        <p class="venue">${venue}</p>
        <div class="publication-links">
            ${linksHtml}
        </div>
    `;
    
    // Insert before the note
    publicationsContent.insertBefore(publicationItem, noteElement);
}

// Example usage (uncomment and modify as needed):
/*
// Add a new news item
addNewsItem('April 2024', 'New paper submitted to Conference XYZ!');

// Add a new publication
addPublication(
    'My Research Paper Title',
    '<strong>Matthew Zhang</strong>, Co-Author Name',
    'Conference Name 2024',
    {
        'Paper': 'https://example.com/paper.pdf',
        'Code': 'https://github.com/username/repo',
        'Dataset': 'https://example.com/dataset'
    }
);
*/