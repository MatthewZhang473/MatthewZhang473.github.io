# Matthew Zhang - Academic Homepage

A clean, professional academic homepage built for GitHub Pages. This website serves as a central hub for academic work, publications, and professional updates.

## 🌟 Features

- **Clean Academic Design**: Professional layout with academic styling
- **Responsive**: Mobile-friendly design that works on all devices
- **Easy Content Management**: Simple HTML structure for easy updates
- **Resume/CV Section**: Dedicated area for downloadable documents
- **Publications**: Organized display of research work with links
- **News & Updates**: Timeline of recent academic activities
- **Contact Information**: Professional contact details and social links
- **Navigation**: Smooth scrolling navigation with active link highlighting

## 📁 File Structure

```
├── index.html          # Main homepage
├── styles.css          # CSS styling
├── script.js           # JavaScript functionality
├── files/              # Directory for uploads (resume, CV, etc.)
│   └── README.md      # Instructions for file uploads
└── README.md          # This file
```

## 🚀 Getting Started

### Customizing Your Homepage

1. **Update Personal Information**:
   - Edit `index.html` to replace placeholder text with your information
   - Update the header section with your name, title, and affiliation
   - Modify the About section with your research interests

2. **Add Your Resume/CV**:
   - Place your PDF files in the `files/` directory
   - Name them `resume.pdf` and `cv.pdf` 
   - Or update the links in the HTML to match your file names

3. **Update Publications**:
   - Replace sample publications with your actual work
   - Add links to papers, code repositories, datasets, etc.
   - Use the provided CSS classes for consistent styling

4. **Manage News & Updates**:
   - Add new announcements, paper acceptances, awards
   - Keep the chronological order (newest first)
   - Remove the placeholder entries

5. **Update Contact Information**:
   - Replace placeholder email, office, and address
   - Update social media and professional profile links

### Local Development

To test changes locally:

```bash
# Navigate to the project directory
cd MatthewZhang473.github.io

# Start a local server
python3 -m http.server 8000

# Open http://localhost:8000 in your browser
```

### GitHub Pages Deployment

This site is automatically deployed via GitHub Pages:
- The site will be available at `https://MatthewZhang473.github.io`
- Changes pushed to the main branch are automatically deployed
- No additional configuration needed

## 🎨 Customization

### Colors
The main color scheme uses:
- Header: Blue gradient (`#2c3e50` to `#3498db`)
- Navigation: Dark blue (`#34495e`)
- Accent: Blue (`#3498db`)
- Text: Dark gray (`#333`)

### Fonts
- Headers: 'Crimson Text' (serif)
- Body: 'Source Sans Pro' (sans-serif)

### Adding Content

**New Publication:**
```html
<div class="publication-item">
    <h3>Your Paper Title</h3>
    <p class="authors"><strong>Matthew Zhang</strong>, Co-authors</p>
    <p class="venue">Conference/Journal Name, Year</p>
    <div class="publication-links">
        <a href="link-to-paper.pdf" class="pub-link">Paper</a>
        <a href="link-to-code" class="pub-link">Code</a>
    </div>
</div>
```

**New News Item:**
```html
<div class="news-item">
    <span class="date">Month Year</span>
    <p>Your news content here.</p>
</div>
```

## 📱 Mobile Responsiveness

The site includes responsive design features:
- Collapsible navigation on mobile
- Flexible layouts for all screen sizes
- Optimized typography for readability

## 🔧 Technical Details

- **No build process**: Simple HTML/CSS/JS for easy maintenance
- **SEO friendly**: Proper meta tags and semantic HTML
- **Accessibility**: ARIA labels and semantic structure
- **Print styles**: Optimized for printing/PDF generation

## 📄 License

This project is open source. Feel free to use it as a template for your own academic homepage.

## 🤝 Contributing

This is a personal academic homepage, but suggestions for improvements are welcome via issues or pull requests.