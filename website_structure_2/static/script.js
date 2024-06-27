document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded, initializing map...');
    var map = L.map('map').setView([51.505, -0.09], 13);

    // Adding black and white tile layer with labels
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        maxZoom: 19
    }).addTo(map);

    var marker;

    map.on('click', function(e) {
        console.log('Map clicked at:', e.latlng);
        document.getElementById('coordinates').value = e.latlng.lat + ", " + e.latlng.lng;
        if (marker) {
            map.removeLayer(marker);
        }
        marker = new L.Marker(e.latlng).addTo(map);
    });

    L.Control.geocoder().addTo(map);

    function displayImage(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.createElement('img');
            img.src = e.target.result;
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = ''; // Clear previous images
            resultsDiv.appendChild(img);
        };
        reader.readAsDataURL(file);
    }

    document.getElementById('imageInput').addEventListener('change', function() {
        const imageFile = this.files[0];
        displayImage(imageFile);
    });

    function submitData() {
        console.log('Submit button clicked');
        const imageFile = document.getElementById('imageInput').files[0];
        if (!imageFile) {
            console.error('No image file selected');
            document.getElementById('results').innerHTML = '<p>Error: No image file selected</p>';
            return;
        }

        const coordinatesValue = document.getElementById('coordinates').value;
        if (!coordinatesValue) {
            console.error('No coordinates selected');
            document.getElementById('results').innerHTML = '<p>Error: No coordinates selected</p>';
            return;
        }

        const coordinates = coordinatesValue.split(", ");
        const lat = parseFloat(coordinates[0]);
        const lon = parseFloat(coordinates[1]);
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('lat', lat);
        formData.append('lon', lon);

        console.log('Submitting data', { lat, lon, imageFile });

        fetch('/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Response received:', response);
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            console.log('JSON data received:', data);

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = ''; // Clear previous results

            if (data.segmented_image_path) {
                const segmentedImg = document.createElement('img');
                segmentedImg.src = data.segmented_image_path;
                resultsDiv.appendChild(segmentedImg);
            }

            if (data.classified_image_path) {
                const classifiedImg = document.createElement('img');
                classifiedImg.src = data.classified_image_path;
                resultsDiv.appendChild(classifiedImg);
            }

            if (data.building_info) {
                const buildingInfo = document.createElement('div');
                buildingInfo.innerHTML = `
                    <p>OSM ID: ${data.building_info.osm_id}</p>
                    <p>Building Name: ${data.building_info.building_name}</p>
                    <p>Building Category: ${data.building_info.building_category}</p>
                    <p>Building Height: ${data.building_info.building_height} meters</p>
                    <p>Building Height Source: ${data.building_info.height_source}</p>
                    <p>Footprint Area: ${data.building_info.footprint_area} square meters</p>
                    <p>Facade Area: ${data.building_info.facade_area} square meters</p>
                    <p>Material Areas:</p>
                    <ul>
                        ${Object.entries(data.building_info.material_areas).map(([material, area]) => `<li>${material}: ${area} square meters</li>`).join('')}
                    </ul>
                `;
                resultsDiv.appendChild(buildingInfo);
            }

            if (data.message) {
                const message = document.createElement('p');
                message.innerText = data.message;
                resultsDiv.appendChild(message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('results').innerHTML = `<p>Error: ${error.message}</p>`;
        });
    }

    document.getElementById('submitButton').addEventListener('click', submitData);
    console.log('Submit button event listener attached...');

    // Add event listener for reset button
    document.getElementById('resetButton').addEventListener('click', function() {
        console.log('Reset button clicked');
        document.getElementById('coordinates').value = '';
        document.getElementById('imageInput').value = '';
        document.getElementById('results').innerHTML = '';
        if (marker) {
            map.removeLayer(marker);
        }
        map.setView([51.505, -0.09], 13); // Reset map view to initial position
    });
});
