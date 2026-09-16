document.addEventListener('DOMContentLoaded', () => {
    let selectedSamplePath = null;

    // Tab Switching Logic
    const tabBtns = document.querySelectorAll('.nav-tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPaneId = btn.getAttribute('data-tab');
            document.getElementById(targetPaneId).classList.add('active');

            if (targetPaneId === 'tab-metrics') {
                loadMetrics();
            }
        });
    });

    // Dataset Subset Selector Change
    const datasetSelect = document.getElementById('dataset_type_select');
    if (datasetSelect) {
        datasetSelect.addEventListener('change', (e) => {
            loadSampleQueries(e.target.value);
        });
        loadSampleQueries(datasetSelect.value);
    }

    // Load Sample Query Thumbnails
    async function loadSampleQueries(subset) {
        const grid = document.getElementById('samples_grid');
        grid.innerHTML = '<div style="grid-column: 1/-1; text-align:center; color:#64748b;">Loading sample queries...</div>';

        try {
            const res = await fetch(`/api/sample_queries?dataset_type=${subset}`);
            const data = await res.json();

            if (data.samples && data.samples.length > 0) {
                grid.innerHTML = '';
                data.samples.forEach((s, idx) => {
                    const img = document.createElement('img');
                    img.src = `/dataset_image/${s.rel_path}`;
                    img.className = 'sample-img-thumb';
                    img.title = s.filename;
                    
                    if (idx === 0) {
                        img.classList.add('selected');
                        selectedSamplePath = s.rel_path;
                        updateQueryPreview(img.src);
                    }

                    img.addEventListener('click', () => {
                        document.querySelectorAll('.sample-img-thumb').forEach(i => i.classList.remove('selected'));
                        img.classList.add('selected');
                        selectedSamplePath = s.rel_path;
                        document.getElementById('custom_file_input').value = ''; // clear upload
                        updateQueryPreview(img.src);
                    });

                    grid.appendChild(img);
                });
            } else {
                grid.innerHTML = '<div style="grid-column: 1/-1; color:#ef4444;">No query images found.</div>';
            }
        } catch (err) {
            grid.innerHTML = '<div style="grid-column: 1/-1; color:#ef4444;">Error loading samples.</div>';
        }
    }

    // Custom File Upload Listener
    const fileInput = document.getElementById('custom_file_input');
    const uploadBox = document.getElementById('upload_box');

    if (uploadBox && fileInput) {
        uploadBox.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    updateQueryPreview(event.target.result);
                    selectedSamplePath = null;
                    document.querySelectorAll('.sample-img-thumb').forEach(i => i.classList.remove('selected'));
                };
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }

    function updateQueryPreview(src) {
        const previewImg = document.getElementById('query_preview_img');
        if (previewImg) previewImg.src = src;
    }

    // Submit Re-ID Form Query
    const queryBtn = document.getElementById('run_reid_btn');
    if (queryBtn) {
        queryBtn.addEventListener('click', runReIDQuery);
    }

    async function runReIDQuery() {
        const spinner = document.getElementById('results_spinner');
        const matchesGrid = document.getElementById('matches_grid');
        const resultsContainer = document.getElementById('results_container');
        const heatmapImg = document.getElementById('heatmap_preview_img');

        spinner.style.display = 'block';
        matchesGrid.innerHTML = '';
        resultsContainer.style.display = 'none';

        const formData = new FormData();
        const datasetType = document.getElementById('dataset_type_select').value;
        const topK = document.getElementById('top_k_select').value;

        formData.append('dataset_type', datasetType);
        formData.append('top_k', topK);

        if (fileInput.files && fileInput.files[0]) {
            formData.append('file', fileInput.files[0]);
        } else if (selectedSamplePath) {
            formData.append('sample_rel_path', selectedSamplePath);
        } else {
            alert('Please select a sample query image or upload an image.');
            spinner.style.display = 'none';
            return;
        }

        try {
            const res = await fetch('/api/reid_query', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            spinner.style.display = 'none';

            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }

            resultsContainer.style.display = 'block';
            if (heatmapImg && data.heatmap_url) {
                heatmapImg.src = data.heatmap_url;
            }

            // Render Top-K Matches
            data.matches.forEach(match => {
                const card = document.createElement('div');
                card.className = 'match-card';
                card.innerHTML = `
                    <div class="rank-badge">#${match.rank}</div>
                    <img src="${match.web_url}" alt="PID ${match.pid}">
                    <div class="similarity-pill">${match.similarity}% Match</div>
                    <div class="match-meta">PID: <strong>${match.pid}</strong></div>
                    <div class="match-meta">Cam: ${match.camid + 1}</div>
                `;
                matchesGrid.appendChild(card);
            });
        } catch (err) {
            spinner.style.display = 'none';
            alert('Failed to execute Person Re-ID search. Check server logs.');
        }
    }

    // Load Metrics Table & Summary Cards
    async function loadMetrics() {
        const tbody = document.getElementById('metrics_tbody');
        if (!tbody) return;

        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();

            tbody.innerHTML = '';
            data.metrics.forEach(m => {
                let tagClass = 'tag-small';
                if (m.dataset_type === 'with_bag') tagClass = 'tag-bag';
                if (m.dataset_type === 'without_bag') tagClass = 'tag-nobag';
                if (m.dataset_type === 'both_large') tagClass = 'tag-large';

                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><span class="pill-tag ${tagClass}">${m.dataset_type}</span></td>
                    <td>${m.num_query}</td>
                    <td>${m.num_gallery}</td>
                    <td><strong style="color: #34d399;">${m.rank1}%</strong></td>
                    <td>${m.rank5}%</td>
                    <td>${m.rank10}%</td>
                    <td><strong style="color: #60a5fa;">${m.mAP}%</strong></td>
                `;
                tbody.appendChild(row);
            });

            // Update top summary cards with benchmark data
            if (data.metrics.length > 0) {
                const b = data.metrics[0];
                const r1 = document.getElementById('card_rank1');
                const r5 = document.getElementById('card_rank5');
                const r10 = document.getElementById('card_rank10');
                const mapCard = document.getElementById('card_map');
                if (r1) r1.textContent = `${b.rank1}%`;
                if (r5) r5.textContent = `${b.rank5}%`;
                if (r10) r10.textContent = `${b.rank10}%`;
                if (mapCard) mapCard.textContent = `${b.mAP}%`;
            }
        } catch (err) {
            console.error('Metrics loading error:', err);
        }
    }

    // Initial load of metrics
    loadMetrics();
});
