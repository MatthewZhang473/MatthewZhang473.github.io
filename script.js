// Academic Sidebar Theme JavaScript

// 1. Blog Functionality
function loadBlogPost(url, skipHashUpdate = false) {
    const blogList = document.querySelector('.blog-preview-list');
    const blogContent = document.getElementById('blog-post-content');
    const viewer = document.getElementById('markdown-viewer');

    if (!blogList || !blogContent || !viewer) return;

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

function showBlogList() {
    const blogList = document.querySelector('.blog-preview-list');
    const blogContent = document.getElementById('blog-post-content');
    if (blogList && blogContent) {
        blogList.style.display = 'block';
        blogContent.style.display = 'none';
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