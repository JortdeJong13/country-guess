// Messages for high confidence predictions
const highConfidenceMessages = [
  "I'm absolutely sure this is {{guessed}}!",
  "No doubt about it — this is {{guessed}}.",
  "This is definitely {{guessed}} — 100% certain.",
  "I'd bet anything this is {{guessed}}.",
  "{{guessed}} for sure! It’s unmistakable.",
  "This is classic {{guessed}} — crystal clear.",
  "Zero hesitation — this has to be {{guessed}}.",
  "I’d recognize {{guessed}} anywhere. No question.",
  "It’s obvious — this is {{guessed}}.",
  "Undeniably {{guessed}}. Couldn’t be anything else.",
];

// Messages for medium confidence predictions
export const mediumConfidenceMessages = [
  "This looks a lot like {{guessed}} to me.",
  "I think this could be {{guessed}}. What do you think?",
  "This reminds me of {{guessed}}.",
  "My best guess would be {{guessed}}.",
  "It has the shape of {{guessed}}, I’d say.",
  "I'm leaning toward {{guessed}} based on the details.",
  "This feels like {{guessed}}, but I could be off.",
  "I’d guess {{guessed}}, though I’m not entirely sure.",
  "It closely resembles {{guessed}}.",
  "This might be {{guessed}} — the features match.",
];

// Messages for low confidence predictions
export const lowConfidenceMessages = [
  "I'm not sure, but this might be {{guessed}}?",
  "Tough one... maybe {{guessed}}?",
  "Could it be {{guessed}}? I'm guessing here.",
  "I just don't know, I'm guessing {{guessed}}.",
  "Hmm… possibly {{guessed}}?",
  "It’s hard to tell, but I’ll say {{guessed}}.",
  "Best guess: {{guessed}} — though I'm not confident.",
  "I’m unsure, but it slightly resembles {{guessed}}.",
  "Could be {{guessed}}, but that's a stretch.",
];

// Messages for correct predictions
export const correctMessages = [
  "Yep — that’s {{selected}}!",
  "You drew {{selected}} perfectly!",
  "Spot on! That's {{selected}}!",
  "Exactly right — this is {{selected}}.",
  "Awesome job on {{selected}}!",
  "You nailed it — {{selected}} all the way!",
  "This is a great depiction of {{selected}}!",
  "Nice work — I added your {{selected}} to the test set!",
  "That’s a textbook {{selected}}!",
];

// Messages for incorrect predictions
export const incorrectMessages = [
  "Oh! It’s actually {{selected}} — I see why I thought it was {{guessed}}.",
  "Ah, {{selected}}! That does look a lot like {{guessed}} though.",
  "Now I see — it’s {{selected}}, not {{guessed}}.",
  "Oh... Okay, I guess the test set could use a tricky {{selected}}!",
  "I confused {{selected}} with {{guessed}} — good to know for next time!",
  "{{selected}}! That one tripped me up.",
  "Looks like I mistook {{selected}} for {{guessed}}. I'm learning!",
  "I see it now — definitely {{selected}}, not {{guessed}}.",
  "Oops, that’s {{selected}}. I’ll remember the difference.",
  "{{selected}} — got it! I’ll be sharper next time.",
];

const countryFacts = {
  Mexico:
    "Mexico is home to the world's smallest volcano, Cuexcomate, standing at just 13 meters tall.",
  Somalia:
    "Somalia has the longest coastline in mainland Africa, stretching over 3,000 kilometers.",
  "Ivory Coast":
    "The Ivory Coast is the world's largest producer of cocoa beans.",
  Indonesia:
    "Indonesia is made up of over 17,000 islands, making it the world's largest archipelago.",
  "St. Lucia":
    "St. Lucia is home to the Pitons, two volcanic spires and a UNESCO World Heritage site.",
  Poland:
    "Poland contains the Białowieża Forest, one of Europe's last primeval forests.",
  Canada: "Canada has more lakes than the rest of the world's lakes combined.",
  Sweden:
    "Sweden's Icehotel in Jukkasjärvi is rebuilt every year from ice and snow.",
  Reunion:
    "Reunion is home to one of the world's most active volcanoes, Piton de la Fournaise.",
  "Pacific Islands (Palau)":
    "Palau features Jellyfish Lake, where millions of stingless jellyfish migrate daily.",
  Kazakhstan: "Kazakhstan is the largest landlocked country in the world.",
  Lebanon:
    "In Lebanon, you can ski in the morning and swim in the Mediterranean Sea in the afternoon.",
  Switzerland:
    "Switzerland has over 7,000 lakes, with Lake Geneva being the largest.",
  "Sierra Leone":
    "Sierra Leone is home to the Tacugama Chimpanzee Sanctuary, protecting endangered primates.",
  "Trinidad and Tobago":
    "Trinidad contains the world's largest natural deposit of asphalt at Pitch Lake.",
  Bolivia: "Bolivia is home to Salar de Uyuni, the world's largest salt flat.",
  Norway:
    "Norway features the world's longest road tunnel, the Lærdal Tunnel, at 24.5 km.",
  Hungary:
    "Hungary's capital, Budapest, has more thermal springs than any other capital city.",
  Algeria: "Algeria is the largest country in Africa by land area.",
  Iraq: "Iraq is home to the ancient city of Babylon, a UNESCO World Heritage site.",
  Nepal:
    "Nepal contains eight of the world's ten highest peaks, including Mount Everest.",
  Colombia:
    "Colombia is the only South American country with coastlines on both the Pacific and Atlantic Oceans.",
  Bahrain:
    "Bahrain is an archipelago of 33 natural islands in the Persian Gulf.",
  "Bosnia and Herzegovina":
    "Bosnia and Herzegovina is home to the historic Stari Most bridge in Mostar, rebuilt after the war.",
  Uganda:
    "Uganda contains the source of the Nile River, the world's longest river.",
  Uruguay:
    "Uruguay's capital, Montevideo, is the southernmost capital city in South America.",
  Togo: "Togo features a unique lagoon, Lake Togo, separated from the Atlantic by a narrow strip of land.",
  Martinique:
    "Martinique's Mount Pelée erupted in 1902, destroying the city of Saint-Pierre.",
  India: "India is the only country with both lions and tigers in the wild.",
  Egypt:
    "Egypt is home to the Great Pyramid of Giza, the only surviving ancient wonder.",
  "Equatorial Guinea":
    "Equatorial Guinea is the only African country where Spanish is an official language.",
  Uzbekistan:
    "Uzbekistan is one of only two doubly landlocked countries in the world.",
  Greece: "Greece has more than 6,000 islands, though only 227 are inhabited.",
  "Dominican Republic":
    "The Dominican Republic shares the island of Hispaniola with Haiti.",
  Pakistan: "Pakistan is home to K2, the second-highest mountain in the world.",
  Brazil: "Brazil contains about 60% of the Amazon Rainforest.",
  Sudan: "Sudan has more pyramids than any other country, including Egypt.",
  Slovakia: "Slovakia is home to over 6,000 caves, many open to the public.",
  Portugal: "Portugal is the westernmost country of mainland Europe.",
  Macedonia:
    "Macedonia's Lake Ohrid is one of Europe's deepest and oldest lakes.",
  Argentina:
    "Argentina is home to the highest peak in the Americas, Aconcagua.",
  "Solomon Islands":
    "The Solomon Islands consist of nearly 1,000 islands in the South Pacific.",
  Guadeloupe:
    "Guadeloupe is an overseas region of France located in the Caribbean.",
  Mali: "Mali's ancient city of Timbuktu was a center of learning in the 15th century.",
  Albania:
    "Albania has more bunkers per square kilometer than any other country.",
  Ireland: "Ireland is known as the Emerald Isle for its lush green landscape.",
  Australia: "Australia is the only country that is also a continent.",
  Suriname: "Suriname has over 90% of its land covered by tropical rainforest.",
  Spain:
    "Spain has the second-highest number of UNESCO World Heritage Sites in the world.",
  "South Georgia and the South Sandwich Is":
    "South Georgia is home to vast penguin colonies and has no permanent residents.",
  Slovenia:
    "Slovenia has over 60% of its land covered by forest, making it one of Europe's greenest countries.",
  "Falkland Islands (Islas Malvinas)":
    "The Falkland Islands have more sheep than people.",
  Angola: "Angola's Kalandula Falls are among Africa's largest waterfalls.",
  Mauritius:
    "Mauritius was the only known habitat of the now-extinct dodo bird.",
  Kuwait: "Kuwait has no natural rivers or lakes in its territory.",
  Botswana:
    "Botswana's Okavango Delta is one of the world's largest inland river deltas.",
  Azerbaijan:
    "Azerbaijan is home to the world's first oil well, drilled in 1846.",
  "South Africa":
    "South Africa uniquely has three capital cities: Pretoria, Bloemfontein, and Cape Town.",
  Taiwan:
    "Taiwan was home to the world's tallest building from 2004 to 2010, Taipei 101.",
  "Virgin Islands":
    "The Virgin Islands are one of the few U.S. territories where people drive on the left side of the road.",
  "Cape Verde":
    "Cape Verde is an archipelago of 10 volcanic islands in the central Atlantic Ocean.",
  Honduras: "Honduras is home to the ancient Mayan city of Copán.",
  Peru: "Peru's Machu Picchu is one of the New Seven Wonders of the World.",
  Burundi: "Burundi borders Lake Tanganyika, the world's second-deepest lake.",
  Syria:
    "Syria's capital, Damascus, is considered one of the oldest continuously inhabited cities.",
  Malaysia:
    "Malaysia's Petronas Towers were once the world's tallest buildings.",
  Italy: "Italy contains the world's smallest country, Vatican City.",
  Nigeria: "Nigeria has the largest population in Africa.",
  Japan: "Japan consists of over 6,800 islands.",
  Ukraine: "Ukraine is the largest country entirely within Europe.",
  "Gambia, The": "The Gambia is the smallest country on mainland Africa.",
  Zambia:
    "Zambia shares Victoria Falls, one of the world's largest waterfalls, with Zimbabwe.",
  Gabon: "Gabon has nearly 90% of its land covered by rainforest.",
  Romania:
    "Romania is home to the Transfăgărășan, one of the world's most scenic roads.",
  Iran: "Iran has one of the world's oldest continuous major civilizations.",
  Fiji: "Fiji comprises over 330 islands, of which about 110 are inhabited.",
  Vietnam:
    "Vietnam's Ha Long Bay features thousands of limestone islands and islets.",
  Thailand:
    "Thailand is the only Southeast Asian country never colonized by a European power.",
  Bulgaria:
    "Bulgaria is home to the oldest gold treasure in the world, found in Varna.",
  Turkmenistan:
    "Turkmenistan's Darvaza gas crater is known as the 'Door to Hell'.",
  Georgia: "Georgia's Mount Shkhara reaches a height of 5,193 meters.",
  Zaire: "Zaire, now called DR Congo, contains half of Africa's rainforest.",
  Bhutan:
    "Bhutan measures its progress by Gross National Happiness instead of GDP.",
  "United States of America":
    "The United States has the world's largest economy.",
  Greenland: "Greenland is the world's largest island that isn't a continent.",
  Cuba: "Cuba is the largest island in the Caribbean.",
  Qatar: "Qatar has the highest GDP per capita in the world.",
  Bangladesh:
    "Bangladesh contains the world's largest river delta, the Sundarbans.",
  "Czech Republic":
    "The Czech Republic's Prague Castle is the largest ancient castle in the world.",
  "New Zealand":
    "New Zealand has the world's longest place name: Taumatawhakatangihangakoauauotamateaturipukakapikimaungahoronukupokaiwhenuakitanatahu.",
  Haiti:
    "Haiti was the first post-colonial independent black-led nation in the world.",
  Rwanda:
    "Rwanda is known as the 'Land of a Thousand Hills' due to its mountainous terrain.",
  "Netherlands Antilles":
    "The Netherlands Antilles consisted of six Caribbean islands until 2010.",
  Cameroon: "Cameroon's Mount Cameroon is the highest point in West Africa.",
  Philippines: "The Philippines consists of over 7,600 islands.",
  Morocco:
    "Morocco contains part of the world's largest hot desert, the Sahara.",
  "Heard Island & McDonald Islands":
    "These islands are home to Australia's only active volcano, Big Ben.",
  "Papua New Guinea":
    "Papua New Guinea has over 850 indigenous languages, more than any other country.",
  Israel: "Israel's Dead Sea is the lowest point on Earth's surface.",
  "Western Sahara":
    "Western Sahara has the world's largest phosphate reserves.",
  "United Kingdom":
    "The UK's Loch Ness holds more freshwater than all lakes in England and Wales combined.",
  "New Caledonia":
    "New Caledonia is surrounded by the world's second-largest double barrier coral reef.",
  Cambodia:
    "Cambodia is home to Angkor Wat, the largest religious monument in the world.",
  Guyana:
    "Guyana's Kaieteur Falls is one of the world's tallest single-drop waterfalls.",
  Nicaragua:
    "Nicaragua's Lake Nicaragua is the only freshwater lake containing sharks.",
  Russia: "Russia spans 11 time zones, more than any other country.",
  Swaziland:
    "Swaziland, now called Eswatini, is one of Africa's last absolute monarchies.",
  Brunei:
    "Brunei hosts one of the world's largest water villages, Kampong Ayer.",
  Belgium: "Belgium has three official languages: Dutch, French, and German.",
  Austria: "Austria's Alps cover over 60% of the country.",
  Dominica:
    "Dominica contains the world's second-largest hot spring, Boiling Lake.",
  "Bahamas, The":
    "The Bahamas comprises around 700 islands and over 2,000 cays.",
  Ghana: "Ghana's Lake Volta is one of the world's largest man-made lakes.",
  Jordan:
    "Jordan's ancient city of Petra is carved entirely into rose-red sandstone cliffs.",
  Chile:
    "Chile contains the Atacama Desert, the driest non-polar desert on Earth.",
  Laos: "Laos's western border is formed by the mighty Mekong River.",
  Afghanistan:
    "Afghanistan's landscape is dominated by the Hindu Kush mountain range.",
  Venezuela:
    "Venezuela's Angel Falls is the world's highest uninterrupted waterfall.",
  Zimbabwe:
    "Zimbabwe shares Victoria Falls, one of the world's largest waterfalls.",
  Malawi:
    "Malawi's Lake Malawi contains more fish species than any other lake.",
  Germany:
    "Germany's Rhine River is one of Europe's longest and most important waterways.",
  "Sri Lanka": "Sri Lanka is known as the 'Pearl of the Indian Ocean'.",
  "Central African Republic":
    "The Central African Republic lies entirely within the tropical zone.",
  "Sao Tome and Principe":
    "Sao Tome and Principe is Africa's second-smallest country by population.",
  Kenya:
    "Kenya's Great Rift Valley runs through the entire length of the country.",
  "French Guiana":
    "French Guiana hosts the Guiana Space Centre, a major European spaceport.",
  Guam: "Guam is the westernmost territory of the United States.",
  Mozambique: "Mozambique has one of the longest coastlines in Africa.",
  Barbados: "Barbados is the easternmost island in the Caribbean.",
  "Antigua and Barbuda":
    "Antigua and Barbuda boasts 365 beaches, one for each day of the year.",
  France: "France has the most time zones of any country, totaling 12.",
  "West Bank":
    "The West Bank contains many sites sacred to three major religions.",
  Montenegro:
    "Montenegro is home to Europe's deepest canyon, the Tara River Canyon.",
  Netherlands: "The Netherlands has about a third of its land below sea level.",
  "Costa Rica": "Costa Rica contains 5% of the world's biodiversity.",
  "South Korea": "South Korea's Jeju Island is a UNESCO World Heritage site.",
  Tajikistan: "Tajikistan has over 90% of its territory covered by mountains.",
  Liberia: "Liberia is Africa's oldest republic, founded in 1847.",
  Kiribati: "Kiribati is the only country situated in all four hemispheres.",
  "Tanzania, United Republic of":
    "Tanzania's Mount Kilimanjaro is Africa's highest peak.",
  Niger: "Niger has over 80% of its territory covered by the Sahara Desert.",
  Byelarus:
    "Belarus (Byelarus) has Europe's largest remaining primeval forest.",
  Libya: "Libya recorded the hottest temperature ever measured on Earth.",
  "Jan Mayen":
    "Jan Mayen is an Arctic island dominated by the Beerenberg volcano.",
  Cyprus: "Cyprus is divided between Greek and Turkish territories.",
  Turkey:
    "Turkey's largest city, Istanbul, spans two continents: Europe and Asia.",
  Ecuador: "Ecuador has the equator running directly through its territory.",
  Svalbard:
    "Svalbard hosts the Global Seed Vault, preserving seeds from around the world.",
  "Puerto Rico":
    "Puerto Rico contains El Yunque, the only tropical rainforest in the U.S. National Forest System.",
  "Guinea-Bissau":
    "Guinea-Bissau features a complex system of islands and mangroves in the Bijagós Archipelago.",
  Lesotho:
    "Lesotho is the only country entirely above 1,000 meters in elevation.",
  "Man, Isle of":
    "The Isle of Man is located in the Irish Sea and is not part of the UK or EU.",
  "Saudi Arabia":
    "Saudi Arabia contains the world's largest sand desert, the Rub' al Khali.",
  Estonia: "Estonia has over 50% of its territory covered by forest.",
  Mauritania: "Mauritania's Richat Structure is visible from space.",
  Yemen:
    "Yemen's Socotra Island hosts hundreds of plant species found nowhere else on Earth.",
  Oman: "Oman's northern landscape is dominated by the Al Hajar Mountains.",
  Djibouti:
    "Djibouti's Lake Assal is Africa's lowest point at 155m below sea level.",
  Armenia:
    "Armenia's Lake Sevan is one of the largest high-altitude lakes in the world.",
  Ethiopia:
    "Ethiopia's Great Rift Valley splits the country from northeast to southwest.",
  Denmark: "Denmark includes over 400 named islands.",
  Comoros:
    "The Comoros islands were formed by volcanic activity in the Indian Ocean.",
  Eritrea: "Eritrea has one of the longest coastlines along the Red Sea.",
  Croatia: "Croatia boasts over 1,200 islands along its Adriatic coast.",
  Finland:
    "Finland contains around 188,000 lakes, earning it the nickname 'Land of a Thousand Lakes'.",
  Congo: "The Congo River is Africa's second-longest and deepest river.",
  "United Arab Emirates":
    "The UAE is home to the world's tallest building, the Burj Khalifa in Dubai.",
  Senegal: "Senegal is the westernmost country on the African mainland.",
  "Turks and Caicos Islands":
    "The Turks and Caicos Islands are home to one of the largest coral reef systems in the world.",
  "Burkina Faso": "Burkina Faso's terrain is mostly flat savanna.",
  Mongolia: "Mongolia is the least densely populated country in the world.",
  Chad: "Chad's Lake Chad, once Africa's largest lake, has significantly shrunk over the decades.",
  Belize: "Belize hosts the second-largest barrier reef in the world.",
  Tunisia: "Tunisia's landscape is 40% Sahara Desert.",
  "St. Pierre and Miquelon":
    "St. Pierre and Miquelon is the only remaining French territory in North America.",
  Iceland: "Iceland has more than 130 active and inactive volcanoes.",
  "North Korea":
    "North Korea's Mount Paektu is the country's highest and most sacred peak.",
  Vanuatu: "Vanuatu is one of the most seismically active regions on Earth.",
  "El Salvador":
    "El Salvador is known as the 'Land of Volcanoes' with over 20 active ones.",
  Panama: "Panama's famous canal connects the Atlantic and Pacific Oceans.",
  Guatemala: "Guatemala has 37 volcanoes, with several still active.",
  "Faroe Islands":
    "The Faroe Islands have no naturally growing trees due to strong winds.",
  Madagascar:
    "Madagascar is the world's fourth-largest island and 90% of its wildlife is found nowhere else.",
  China:
    "China shares the world's highest mountain, Mount Everest, with Nepal.",
  Kyrgyzstan: "Kyrgyzstan has over 90% of its territory covered in mountains.",
  Luxembourg:
    "Luxembourg is one of the smallest and wealthiest countries in the world.",
  Serbia:
    "Serbia's capital, Belgrade, is built where the Danube River meets the Sava.",
  Jamaica:
    "Jamaica's Blue Mountains are famous for their coffee and scenic views.",
  Lithuania:
    "Lithuania contains the geographic center of Europe near its capital, Vilnius.",
  Moldova:
    "Moldova's landscape is characterized by gently rolling hills and numerous vineyards.",
  Benin:
    "Benin has a narrow Atlantic coastline stretching just 121 kilometers.",
  "Myanmar (Burma)":
    "Myanmar's Irrawaddy River is crucial for transportation and agriculture.",
  Namibia:
    "Namibia contains the Namib Desert, one of the oldest deserts in the world.",
  Paraguay:
    "Paraguay is one of only two landlocked countries in South America.",
  "Western Samoa":
    "Western Samoa is one of the first countries to see each new day due to its location.",
  Guinea: "Guinea's highlands are the source of the Niger River.",
  Latvia: "Latvia has over half of its territory covered by forest.",
};

export function getConfidenceBasedMessage(score, guessedCountry) {
  const messageList =
    score > 0.28
      ? highConfidenceMessages
      : score > 0.18
        ? mediumConfidenceMessages
        : lowConfidenceMessages;
  return getRandomMessage(messageList, guessedCountry, guessedCountry);
}

export function getRandomMessage(messageList, guessedCountry, selectedCountry) {
  // Pick a random message from the list
  const randomIndex = Math.floor(Math.random() * messageList.length);
  const template = messageList[randomIndex];

  // Replace placeholders with the given values
  return template
    .replace("{{selected}}", selectedCountry)
    .replace("{{guessed}}", guessedCountry);
}

export function getCountryFacts(country) {
  const funFact = countryFacts[country]
    ? `\n\nFun fact: ${countryFacts[country]}`
    : "";
  return funFact;
}
