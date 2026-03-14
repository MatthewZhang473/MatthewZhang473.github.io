// Academic Sidebar Theme JavaScript

// 1. Blog Functionality
let currentPostUrl = '';
let activeHeadingObserver;

function slugifyHeading(text) {
    return text
        .toLowerCase()
        .trim()
        .replace(/[^\w\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-');
}

function scrollToHeading(id) {
    const target = document.getElementById(id);
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildBlogNavigation(viewer) {
    if (!viewer) return;

    const toc = document.getElementById('blog-toc');
    const tocNav = document.getElementById('blog-toc-nav');
    if (!toc || !tocNav) return;

    const headings = Array.from(viewer.querySelectorAll('h2, h3'));
    const slugCounts = new Map();

    const getHeadingMarkup = heading => {
        const clone = heading.cloneNode(true);
        clone.querySelectorAll('.heading-anchor').forEach(node => node.remove());
        clone.querySelectorAll('.katex-mathml').forEach(node => node.remove());
        return clone.innerHTML.trim();
    };

    headings.forEach(heading => {
        const plainText = heading.textContent.trim();
        const baseSlug = slugifyHeading(plainText) || 'section';
        const count = slugCounts.get(baseSlug) || 0;
        slugCounts.set(baseSlug, count + 1);

        const slug = count === 0 ? baseSlug : `${baseSlug}-${count + 1}`;
        heading.id = slug;
        heading.classList.add('anchored-heading');

        const anchor = document.createElement('a');
        anchor.href = '#';
        anchor.className = 'heading-anchor';
        anchor.setAttribute('aria-label', `Scroll to ${plainText}`);
        anchor.textContent = '#';
        anchor.addEventListener('click', event => {
            event.preventDefault();
            scrollToHeading(slug);
        });
        heading.appendChild(anchor);
    });

    tocNav.innerHTML = '';
    if (headings.length === 0) {
        toc.style.display = 'none';
        return;
    }

    toc.style.display = 'block';
    const list = document.createElement('ol');
    list.className = 'blog-toc-list';

    headings.forEach(heading => {
        const item = document.createElement('li');
        item.className = `toc-${heading.tagName.toLowerCase()}`;

        const link = document.createElement('a');
        link.href = '#';
        link.dataset.target = heading.id;
        link.innerHTML = getHeadingMarkup(heading);
        link.addEventListener('click', event => {
            event.preventDefault();
            scrollToHeading(heading.id);
        });

        item.appendChild(link);
        list.appendChild(item);
    });

    tocNav.appendChild(list);

    if (activeHeadingObserver) {
        activeHeadingObserver.disconnect();
    }

    activeHeadingObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            const link = tocNav.querySelector(`[data-target="${entry.target.id}"]`);
            if (!link) return;

            if (entry.isIntersecting) {
                tocNav.querySelectorAll('a').forEach(anchor => anchor.classList.remove('active'));
                link.classList.add('active');
            }
        });
    }, {
        rootMargin: '-20% 0px -65% 0px',
        threshold: 0
    });

    headings.forEach(heading => activeHeadingObserver.observe(heading));

    const referenceHeading = headings.find(heading =>
        heading.textContent.replace(/#$/, '').trim().toLowerCase() === 'references'
    );

    if (referenceHeading) {
        const referenceList = referenceHeading.nextElementSibling;
        if (referenceList && referenceList.tagName === 'OL') {
            referenceList.classList.add('reference-list');
            Array.from(referenceList.children).forEach((item, index) => {
                item.classList.add('reference-item');
                item.dataset.referenceNumber = `${index + 1}`;
            });
        }
    }
}

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

            buildBlogNavigation(viewer);

            // 3. Render Mermaid Diagrams
            const mermaidBlocks = viewer.querySelectorAll('code.language-mermaid');
            mermaidBlocks.forEach(block => {
                const pre = block.parentElement;
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = block.textContent;
                pre.replaceWith(div);
            });
            if (window.mermaid) {
                mermaid.run({
                    nodes: viewer.querySelectorAll('.mermaid')
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
        if (activeHeadingObserver) {
            activeHeadingObserver.disconnect();
        }
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
