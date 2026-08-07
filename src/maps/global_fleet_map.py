import pandas as pd
import folium
import json
import requests

df = pd.read_csv("data/processed/global_fleet.csv")
df = df[df['type_navire'] == "Flotte totale"].copy()

excluded_regions = [
    "Monde", "Asie", "Océanie", "Europe", "Amérique", "Amérique du Nord", 
    "Amérique du Sud", "Amérique centrale", "Afrique", "Antarctique",
    "BRICS", "G20", "G77", "OCDE", "PMA", "SIDS", "LLDCs", "Union européenne",
    "Union europeenne (2020 …)", "G-77 (Groupe des 77)", "G20 (Groupe des Vingt)",
    "OCDE (Organisation de cooperation et de developpement economiques)",
    "PMA (Pays les moins avances)", "PMA : Afrique", "PMA : Asie", "PMA : Iles et Haiti",
    "SIDS (Petits Etats insulaires en developpement) (UN-OHRLLS)",
    "SIDS : Atlantique et ocean Indien", "SIDS : Caraibes", "SIDS : Pacifique",
    "LLDCs (Pays en developpement sans littoral)",
    "Economies developpees", "Economies developpees : Ameriques",
    "Economies developpees : Asie et Oceanie", "Economies developpees : Europe",
    "Economies en developpement", "Economies en developpement : Afrique",
    "Economies en developpement : Ameriques", "Economies en developpement : Asie et Oceanie",
    "Economies en developpement sans la Chine", "Economies en developpement sans les PMA",
    "Europe, Amerique du Nord, Australie et Nouvelle-Zelande",
    "Amerique septentrionale et Europe",
    "Asie centrale et meridionale", "Asie et Oceanie", 
    "Asie occidentale et Afrique septentrionale", "Asie orientale et Asie du Sud-Est",
    
]
df = df[~df['economie'].isin(excluded_regions)]


# 2. MAPPING COMPLET (FRANÇAIS -> ANGLAIS GEOJSON)
country_mapping = {
    "Afrique du Sud": "South Africa",
    "Algerie": "Algeria",
    "Angola": "Angola",
    "Benin": "Benin",
    "Botswana": "Botswana",
    "Burkina Faso": "Burkina Faso",
    "Burundi": "Burundi",
    "Cabo Verde": "Cape Verde",
    "Cameroun": "Cameroon",
    "Comores": "Comoros",
    "Congo": "Republic of the Congo",
    "Rep. dem. du Congo": "Democratic Republic of the Congo",
    "Cote d'Ivoire": "Ivory Coast",
    "Djibouti": "Djibouti",
    "Egypte": "Egypt",
    "Erythree": "Eritrea",
    "Eswatini": "Eswatini",
    "Ethiopie": "Ethiopia",
    "Gabon": "Gabon",
    "Gambie": "Gambia",
    "Ghana": "Ghana",
    "Guinee": "Guinea",
    "Guinee equatoriale": "Equatorial Guinea",
    "Guinee-Bissau": "Guinea-Bissau",
    "Kenya": "Kenya",
    "Lesotho": "Lesotho",
    "Liberia": "Liberia",
    "Libye": "Libya",
    "Madagascar": "Madagascar",
    "Malawi": "Malawi",
    "Mali": "Mali",
    "Maurice": "Mauritius",
    "Mauritanie": "Mauritania",
    "Maroc": "Morocco",
    "Mozambique": "Mozambique",
    "Namibie": "Namibia",
    "Niger": "Niger",
    "Nigeria": "Nigeria",
    "Ouganda": "Uganda",
    "Rwanda": "Rwanda",
    "Sao Tome-et-Principe": "Sao Tome and Principe",
    "Senegal": "Senegal",
    "Seychelles": "Seychelles",
    "Sierra Leone": "Sierra Leone",
    "Somalie": "Somaliland",
    "Somalie": "Somalia",
    "Soudan du Sud": "South Sudan",
    "Soudan": "Sudan",
    "Republique-Unie de Tanzanie": "Tanzania",
    "Tchad": "Chad",
    "Togo": "Togo",
    "Tunisie": "Tunisia",
    "Zambie": "Zambia",
    "Zimbabwe": "Zimbabwe",
    
    "Antigua-et-Barbuda": "Antigua and Barbuda",
    "Argentine": "Argentina",
    "Bahamas": "Bahamas",
    "Barbade": "Barbados",
    "Belize": "Belize",
    "Bolivie (Etat plurinational de)": "Bolivia",
    "Bresil": "Brazil",
    "Canada": "Canada",
    "Chili": "Chile",
    "Colombie": "Colombia",
    "Costa Rica": "Costa Rica",
    "Cuba": "Cuba",
    "Dominique": "Dominica",
    "Republique dominicaine": "Dominican Republic",
    "Equateur": "Ecuador",
    "El Salvador": "El Salvador",
    "Etats-Unis": "USA",
    "Grenade": "Grenada",
    "Guatemala": "Guatemala",
    "Guyana": "Guyana",
    "Haiti": "Haiti",
    "Honduras": "Honduras",
    "Jamaique": "Jamaica",
    "Mexique": "Mexico",
    "Nicaragua": "Nicaragua",
    "Panama": "Panama",
    "Paraguay": "Paraguay",
    "Perou": "Peru",
    "Saint-Kitts-et-Nevis": "Saint Kitts and Nevis",
    "Sainte-Lucie": "Saint Lucia",
    "Saint-Vincent-et-les Grenadines": "Saint Vincent and the Grenadines",
    "Suriname": "Suriname",
    "Trinite-et-Tobago": "Trinidad and Tobago",
    "Uruguay": "Uruguay",
    "Venezuela (Rep. bolivarienne du)": "Venezuela",
    "Groenland": "Greenland",
    "Porto Rico": "Puerto Rico",
    "Guam": "Guam",
    "Iles Vierges americaines": "United States Virgin Islands",
    "Iles Vierges britanniques": "British Virgin Islands",
    "Anguilla": "Anguilla",
    "Bermudes": "Bermuda",
    "Iles Caimanes": "Cayman Islands",
    "Iles Falkland (Malvinas)": "Falkland Islands",
    "Curacao": "Curacao",
    "Aruba": "Aruba",
    
    "Federation de Russie":"Russia",
    "Afghanistan": "Afghanistan",
    "Arabie saoudite": "Saudi Arabia",
    "Armenie": "Armenia",
    "Azerbaidjan": "Azerbaijan",
    "Bahrein": "Bahrain",
    "Bangladesh": "Bangladesh",
    "Bhoutan": "Bhutan",
    "Brunei Darussalam": "Brunei",
    "Cambodge": "Cambodia",
    "Chine": "China",
    "Chine, Province de Taiwan": "Taiwan",
    "Chine, RAS de Hong Kong": "Hong Kong",
    "Chine, RAS de Macao": "Macau",
    "Coree du Nord": "North Korea",
    "Republique de Coree": "South Korea",
    "Emirats arabes unis": "United Arab Emirates",
    "Georgie": "Georgia",
    "Inde": "India",
    "Indonesie": "Indonesia",
    "Iran (Republique islamique d')": "Iran",
    "Iraq": "Iraq",
    "Israel": "Israel",
    "Japon": "Japan",
    "Jordanie": "Jordan",
    "Kazakhstan": "Kazakhstan",
    "Kirghizistan": "Kyrgyzstan",
    "Koweit": "Kuwait",
    "Rep. dem. populaire lao": "Laos",
    "Liban": "Lebanon",
    "Malaisie": "Malaysia",
    "Maldives": "Maldives",
    "Mongolie": "Mongolia",
    "Myanmar": "Myanmar",
    "Nepal": "Nepal",
    "Oman": "Oman",
    "Ouzbekistan": "Uzbekistan",
    "Pakistan": "Pakistan",
    "Palaos": "Palau",
    "Palestine": "Palestine",
    "Philippines": "Philippines",
    "Qatar": "Qatar",
    "Republique arabe syrienne": "Syria",
    "Singapour": "Singapore",
    "Sri Lanka": "Sri Lanka",
    "Tadjikistan": "Tajikistan",
    "Thailande": "Thailand",
    "Timor-Leste": "Timor-Leste",
    "Turkiye": "Turkey",
    "Turkmenistan": "Turkmenistan",
    "Viet Nam": "Vietnam",
    "Yemen": "Yemen",
    
    "Albanie": "Albania",
    "Allemagne": "Germany",
    "Andorre": "Andorra",
    "Autriche": "Austria",
    "Belarus": "Belarus",
    "Belgique": "Belgium",
    "Bosnie-Herzegovine": "Bosnia and Herzegovina",
    "Bulgarie": "Bulgaria",
    "Chypre": "Cyprus",
    "Croatie": "Croatia",
    "Danemark": "Denmark",
    "Espagne": "Spain",
    "Estonie": "Estonia",
    "Finlande": "Finland",
    "France": "France",
    "Grece": "Greece",
    "Hongrie": "Hungary",
    "Irlande": "Ireland",
    "Islande": "Iceland",
    "Italie": "Italy",
    "Lettonie": "Latvia",
    "Lituanie": "Lithuania",
    "Luxembourg": "Luxembourg",
    "Macedoine du Nord": "North Macedonia",
    "Malte": "Malta",
    "Moldavie": "Moldova",
    "Monaco": "Monaco",
    "Montenegro": "Montenegro",
    "Norvege": "Norway",
    "Pays-Bas": "Netherlands",
    "Pays-Bas (Royaume des)": "Netherlands",
    "Pologne": "Poland",
    "Portugal": "Portugal",
    "Republique de Moldova": "Moldova",
    "Tchequie": "Czech Republic",
    "Republique tcheque": "Czech Republic",
    "Roumanie": "Romania",
    "Royaume-Uni": "England",
    "Saint-Marin": "San Marino",
    "Serbie": "Serbia",
    "Slovaquie": "Slovakia",
    "Slovenie": "Slovenia",
    "Suede": "Sweden",
    "Suisse": "Switzerland",
    "Ukraine": "Ukraine",
    "Vatican": "Vatican City",
    "Ile de Man": "Isle of Man",
    "Gibraltar": "Gibraltar",
    "Iles Feroe": "Faroe Islands",
    
    "Australie": "Australia",
    "Fidji": "Fiji",
    "Kiribati": "Kiribati",
    "Iles Marshall": "Marshall Islands",
    "Micronesie (Etats federes de)": "Micronesia",
    "Nauru": "Nauru",
    "Nouvelle-Caledonie": "New Caledonia",
    "Nouvelle-Zelande": "New Zealand",
    "Nioue": "Niue",
    "Papouasie-Nouvelle-Guinee": "Papua New Guinea",
    "Polynesie francaise": "French Polynesia",
    "Iles Salomon": "Solomon Islands",
    "Samoa": "Samoa",
    "Tonga": "Tonga",
    "Tuvalu": "Tuvalu",
    "Vanuatu": "Vanuatu",
    "Iles Wallis-et-Futuna": "Wallis and Futuna",
    "Iles Mariannes du Nord": "Northern Mariana Islands",
    "Iles Cook": "Cook Islands",
}

df['economie'] = df['economie'].replace(country_mapping)

geojson_url = "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson"
response = requests.get(geojson_url)
world_geo = response.json()

data_by_year = {}
years = sorted(df['annee'].unique())

for year in years:
    data_by_year[str(year)] = {}
    subset = df[df['annee'] == year]
    for _, row in subset.iterrows():
        data_by_year[str(year)][row['economie']] = int(row['nombre_navires'])

geo_to_data_mapping = {}
missing_countries = []

for feature in world_geo['features']:
    geo_name = feature['properties']['name']
    
    if geo_name in data_by_year[str(years[0])]:
        geo_to_data_mapping[geo_name] = geo_name
    else:
        found = False
        for data_country in data_by_year[str(years[0])].keys():
            if geo_name.lower() == data_country.lower():
                geo_to_data_mapping[geo_name] = data_country
                found = True
                break
            elif geo_name.lower() in data_country.lower() or data_country.lower() in geo_name.lower():
                geo_to_data_mapping[geo_name] = data_country
                found = True
                break
        if not found:
            geo_to_data_mapping[geo_name] = geo_name
            missing_countries.append(geo_name)

m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB Positron')

title_html = '''
<h3 align="center" style="font-size:20px; margin-top:5px; font-family: Arial, sans-serif;">
    <b>Évolution de la flotte maritime mondiale (2011 - 2026)</b>
</h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

def get_color(val):
    if val == 0: return '#e8e8e8'
    elif val <= 100: return '#a6cee3'
    elif val <= 500: return '#b2df8a'
    elif val <= 1500: return '#f4d03f'
    elif val <= 5000: return '#f39c12'
    else: return '#e31a1c'

initial_year = str(years[0])

for feature in world_geo['features']:
    country_name = feature['properties']['name']
    data_country = geo_to_data_mapping.get(country_name, country_name)
    val = data_by_year[initial_year].get(data_country, 0)
    
    popup_id = country_name.replace(' ', '_').replace("'", "").replace('-', '_')
    
    popup_html = f'''
    <div id="popup_{popup_id}" class="popup-content">
        <b>{country_name}</b><br>
        Année: <span id="year_{popup_id}" class="year-display">{initial_year}</span><br>
        Navires: <span id="ships_{popup_id}" class="ships-display">{val}</span>
    </div>
    '''
    
    folium.GeoJson(
        feature,
        style_function=lambda x, val=val: {
            'fillColor': get_color(val),
            'color': 'gray',
            'weight': 0.5,
            'fillOpacity': 0.7 if val > 0 else 0.2
        },
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

# Légende
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 160px; 
            background-color: white; border: 1px solid #ccc; border-radius: 5px; 
            padding: 10px; z-index:9999; font-size:12px; font-family: Arial, sans-serif;">
    <b>Nombre de navires</b><br>
    <div style="background: #e8e8e8; padding: 2px;">0</div>
    <div style="background: #a6cee3; padding: 2px;">≤ 100</div>
    <div style="background: #b2df8a; padding: 2px;">≤ 500</div>
    <div style="background: #f4d03f; padding: 2px;">≤ 1500</div>
    <div style="background: #f39c12; padding: 2px;">≤ 5000</div>
    <div style="background: #e31a1c; padding: 2px; color: white;">> 5000</div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

json_data = json.dumps(data_by_year)
json_years = json.dumps([str(y) for y in years])
json_mapping = json.dumps(geo_to_data_mapping)

js_code = f"""
<script>
    var mapData = {json_data};
    var yearList = {json_years};
    var countryMapping = {json_mapping};
    var isPlaying = false;
    var playInterval = null;

    function getColor(val) {{
        if (val == 0) return '#e8e8e8';
        else if (val <= 100) return '#a6cee3';
        else if (val <= 500) return '#b2df8a';
        else if (val <= 1500) return '#f4d03f';
        else if (val <= 5000) return '#f39c12';
        else return '#e31a1c';
    }}

    function updateMap(year) {{
        var yearData = mapData[year] || {{}};
        
        // Mettre à jour les popups
        for (var countryName in countryMapping) {{
            var dataCountry = countryMapping[countryName];
            var ships = yearData[dataCountry] || 0;
            
            var popupId = countryName.replace(/ /g, '_').replace(/'/g, '').replace(/-/g, '_');
            
            var yearElement = document.getElementById('year_' + popupId);
            if (yearElement) yearElement.textContent = year;
            
            var shipsElement = document.getElementById('ships_' + popupId);
            if (shipsElement) shipsElement.textContent = ships;
        }}
        
        // Mettre à jour les couleurs
        var geojsonElements = document.querySelectorAll('.leaflet-interactive');
        geojsonElements.forEach(function(element) {{
            var countryName = null;
            if (element._popup) {{
                var content = element._popup.getContent();
                if (typeof content === 'string') {{
                    var match = content.match(/<b>(.*?)<\\/b>/);
                    if (match) countryName = match[1];
                }}
            }}
            
            if (countryName) {{
                var dataCountry = countryMapping[countryName] || countryName;
                var ships = yearData[dataCountry] || 0;
                var color = getColor(ships);
                if (element.setStyle) {{
                    element.setStyle({{
                        fillColor: color,
                        fillOpacity: ships > 0 ? 0.7 : 0.2
                    }});
                }}
            }}
        }});
        
        document.getElementById('sliderValue').textContent = year;
        document.getElementById('yearSlider').value = year;
    }}

    function createSlider() {{
        var container = document.createElement('div');
        container.style.position = 'fixed';
        container.style.bottom = '30px';
        container.style.left = '50%';
        container.style.transform = 'translateX(-50%)';
        container.style.zIndex = '1000';
        container.style.backgroundColor = 'white';
        container.style.padding = '15px 25px';
        container.style.borderRadius = '10px';
        container.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
        container.style.display = 'flex';
        container.style.alignItems = 'center';
        container.style.gap = '15px';
        container.style.fontFamily = 'Arial, sans-serif';
        container.style.flexWrap = 'wrap';
        container.style.justifyContent = 'center';
        
        var label = document.createElement('label');
        label.htmlFor = 'yearSlider';
        label.style.fontWeight = 'bold';
        label.style.fontSize = '14px';
        label.textContent = '📅 Année :';
        container.appendChild(label);
        
        var slider = document.createElement('input');
        slider.type = 'range';
        slider.id = 'yearSlider';
        slider.min = yearList[0];
        slider.max = yearList[yearList.length - 1];
        slider.value = yearList[0];
        slider.step = '1';
        slider.style.width = '350px';
        slider.style.cursor = 'pointer';
        container.appendChild(slider);
        
        var yearDisplay = document.createElement('span');
        yearDisplay.id = 'sliderValue';
        yearDisplay.style.fontWeight = 'bold';
        yearDisplay.style.fontSize = '18px';
        yearDisplay.style.minWidth = '50px';
        yearDisplay.style.textAlign = 'center';
        yearDisplay.style.color = '#4CAF50';
        yearDisplay.textContent = yearList[0];
        container.appendChild(yearDisplay);
        
        var playBtn = document.createElement('button');
        playBtn.id = 'playBtn';
        playBtn.textContent = '▶ Lecture';
        playBtn.style.padding = '8px 18px';
        playBtn.style.cursor = 'pointer';
        playBtn.style.border = 'none';
        playBtn.style.borderRadius = '5px';
        playBtn.style.backgroundColor = '#4CAF50';
        playBtn.style.color = 'white';
        playBtn.style.fontWeight = 'bold';
        playBtn.style.fontSize = '13px';
        container.appendChild(playBtn);
        
        var resetBtn = document.createElement('button');
        resetBtn.textContent = '⟲ Réinitialiser';
        resetBtn.style.padding = '8px 18px';
        resetBtn.style.cursor = 'pointer';
        resetBtn.style.border = 'none';
        resetBtn.style.borderRadius = '5px';
        resetBtn.style.backgroundColor = '#f44336';
        resetBtn.style.color = 'white';
        resetBtn.style.fontWeight = 'bold';
        resetBtn.style.fontSize = '13px';
        container.appendChild(resetBtn);
        
        document.body.appendChild(container);
        
        slider.addEventListener('input', function() {{
            document.getElementById('sliderValue').textContent = this.value;
        }});
        
        slider.addEventListener('change', function() {{
            updateMap(this.value);
        }});
        
        playBtn.addEventListener('click', function() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = null;
                isPlaying = false;
                this.textContent = '▶ Lecture';
                this.style.backgroundColor = '#4CAF50';
            }} else {{
                isPlaying = true;
                this.textContent = '⏹ Arrêter';
                this.style.backgroundColor = '#ff6b6b';
                
                var currentYear = document.getElementById('yearSlider').value;
                var currentIndex = yearList.indexOf(currentYear);
                if (currentIndex === -1 || currentIndex === yearList.length - 1) {{
                    currentIndex = 0;
                }}
                
                playInterval = setInterval(function() {{
                    var nextIndex = (currentIndex + 1) % yearList.length;
                    var nextYear = yearList[nextIndex];
                    updateMap(nextYear);
                    currentIndex = nextIndex;
                }}, 1500);
            }}
        }});
        
        resetBtn.addEventListener('click', function() {{
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = null;
                isPlaying = false;
                playBtn.textContent = '▶ Lecture';
                playBtn.style.backgroundColor = '#4CAF50';
            }}
            updateMap(yearList[0]);
        }});
    }}
    
    setTimeout(function() {{
        createSlider();
        setTimeout(function() {{
            updateMap(yearList[0]);
        }}, 200);
    }}, 500);
</script>
"""
m.get_root().html.add_child(folium.Element(js_code))


output_file = "maps/fleet_maps/global_fleet_map.html"
m.save(output_file)
