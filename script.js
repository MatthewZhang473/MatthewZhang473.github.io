// Academic Sidebar Theme JavaScript

// 1. Blog Functionality
let currentPostUrl = '';

function loadBlogPost(url, skipHashUpdate = false) {
    const blogList = document.querySelector('.blog-preview-list');
    const blogContent = document.getElementById('blog-post-content');
    const viewer = document.getElementById('markdown-viewer');
    const sidebar = document.querySelector('.sidebar');
    const wrapper = document.querySelector('.wrapper');

    if (!blogList || !blogContent || !viewer) return;

    currentPostUrl = url;
    const isCN = url.endsWith('_cn.md');
    
    // Update language buttons UI
    const enBtn = document.getElementById('lang-en');
    const cnBtn = document.getElementById('lang-cn');
    if (enBtn && cnBtn) {
        if (isCN) {
            cnBtn.classList.add('active');
            enBtn.classList.remove('active');
        } else {
            enBtn.classList.add('active');
            cnBtn.classList.remove('active');
        }
    }

    // Add a cache-busting timestamp to ensure we get the latest content
    const fetchUrl = `${url}?t=${new Date().getTime()}`;

    fetch(fetchUrl)
        .then(response => {
            if (!response.ok) throw new Error('Failed to load post');
            return response.text();
        })
        .then(markdown => {
            // Render Markdown
            viewer.innerHTML = marked.parse(markdown);
            
            // 2. Render LaTeX using KaTeX (after markdown is parsed)
            if (window.renderMathInElement) {
                renderMathInElement(viewer, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\(', right: '\\)', display: false},
                        {left: '\\[', right: '\\]', display: true}
                    ],
                    throwOnError: false
                });
            }

            // Switch views
            blogList.style.display = 'none';
            blogContent.style.display = 'block';
            
            // Hide sidebar and expand wrapper
            if (sidebar) sidebar.style.display = 'none';
            if (wrapper) wrapper.style.maxWidth = '900px';

            if (!skipHashUpdate) {
                window.location.hash = url;
            }

            window.scrollTo({ top: 0, behavior: 'smooth' });
        })
        .catch(err => {
            console.error(err);
            viewer.innerHTML = '<p style="color:red;">Error loading blog post. Please try again later.</p>';
            blogList.style.display = 'none';
            blogContent.style.display = 'block';
        });
}

function changeLanguage(lang) {
    if (!currentPostUrl) return;

    let newUrl;
    if (lang === 'cn') {
        if (currentPostUrl.endsWith('_cn.md')) return;
        newUrl = currentPostUrl.replace('.md', '_cn.md');
    } else {
        if (!currentPostUrl.endsWith('_cn.md')) return;
        newUrl = currentPostUrl.replace('_cn.md', '.md');
    }

    loadBlogPost(newUrl);
}

function showBlogList() {
    const blogList = document.querySelector('.blog-preview-list');
    const blogContent = document.getElementById('blog-post-content');
    const sidebar = document.querySelector('.sidebar');
    const wrapper = document.querySelector('.wrapper');

    if (blogList && blogContent) {
        blogList.style.display = 'block';
        blogContent.style.display = 'none';
        
        // Restore sidebar and wrapper
        if (sidebar) sidebar.style.display = 'flex';
        if (wrapper) wrapper.style.maxWidth = '1100px';

        currentPostUrl = '';
        window.location.hash = '';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

window.addEventListener('load', () => {
    if (window.location.hash && window.location.hash.startsWith('#blog/')) {
        const postUrl = window.location.hash.substring(1);
        loadBlogPost(postUrl, true);
    }
});

// Handle browser back/forward buttons
window.addEventListener('hashchange', () => {
    if (window.location.hash && window.location.hash.startsWith('#blog/')) {
        const postUrl = window.location.hash.substring(1);
        loadBlogPost(postUrl, true);
    } else if (window.location.hash === '') {
        showBlogList();
    }
});