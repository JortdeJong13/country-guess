import { checkDailyChallenge } from "./daily_challenge.js";

// Messages for high confidence predictions
const highConfidenceMessages = [
  "I'm absolutely sure this is {{guessed}}!",
  "No doubt about it, this is {{guessed}}.",
  "This is definitely {{guessed}}. I'm certain.",
  "I'd bet anything this is {{guessed}}.",
  "{{guessed}} for sure! It’s unmistakable.",
  "This is classic {{guessed}}, crystal clear.",
  "Zero hesitation. This has to be {{guessed}}.",
  "I’d recognize {{guessed}} anywhere. No question.",
  "It’s obvious. This is {{guessed}}.",
  "Undeniably {{guessed}}. Couldn't be anything else.",
];

// Messages for medium confidence predictions
const mediumConfidenceMessages = [
  "This looks a lot like {{guessed}} to me.",
  "I think this could be {{guessed}}. Right?",
  "This reminds me of {{guessed}}.",
  "My best guess would be {{guessed}}.",
  "It has the shape of {{guessed}}, I’d say.",
  "I'm leaning toward {{guessed}} based on the details.",
  "This feels like {{guessed}}, but I could be off.",
  "I’d guess {{guessed}}, though I’m not entirely sure.",
  "It closely resembles {{guessed}}.",
  "This might be {{guessed}}, the features match.",
];

// Messages for low confidence predictions
const lowConfidenceMessages = [
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

const easterEggMessages = [
  "It looks like you have drawn Narnia!",
  "You found the Easter egg, congratulations...",
  "I'm running out of patience. You need to draw a country first.",
  "Are you trying to summon a country?",
  "I'll inform the UN about this new invisible country!",
  "You've drawn... absolutely nothing. A bold choice!",
  "This must be the mythical land of Nowhere!",
  "This must be the lost city of Atlantis!",
];

// Messages for correct predictions
const correctMessages = [
  "Yep, that’s {{selected}}!",
  "You drew {{selected}} perfectly!",
  "Spot on! That's {{selected}}!",
  "Exactly right, this is {{selected}}.",
  "Awesome job on {{selected}}!",
  "You nailed it, {{selected}} all the way!",
  "This is a great depiction of {{selected}}!",
  "Nice work, I added your {{selected}} to the test set!",
  "That’s a textbook {{selected}}!",
];

// Messages for incorrect predictions
const incorrectMessages = [
  "Oh! It’s actually {{selected}}. I see why I thought it was {{guessed}}.",
  "Ah, {{selected}}! That does look a lot like {{guessed}} though.",
  "Now I see, it’s {{selected}}, not {{guessed}}.",
  "Oh... Okay, I guess the test set could use a tricky {{selected}}!",
  "I confused {{selected}} with {{guessed}}. Good to know for next time!",
  "{{selected}}! That one tripped me up.",
  "Looks like I mistook {{selected}} for {{guessed}}. I'm learning!",
  "I see it now, definitely {{selected}}, not {{guessed}}.",
  "Oops, that’s {{selected}}. I’ll remember the difference.",
  "{{selected}}, got it! I’ll be sharper next time.",
];

// Messages for correct user drawings with high confidence
const leaderboardMessagesHigh = [
  "This is my favorite drawing of {{selected}}!",
  "Here someone drew a perfect {{selected}}.",
  "Another drawing of {{selected}}.",
  "Someone drew a nice {{selected}}!",
  "Very clear drawing of {{selected}}.",
  "A correct and recognizable {{selected}} drawing.",
  "Strong details make this clearly {{selected}}.",
  "A very nice submission of {{selected}} from Anonymous42.",
  "Picasso submitted this {{selected}}.",
  "I present to you {{selected}}.",
];

// Messages for correct user drawings with low confidence
const leaderboardMessagesLow = [
  "A simple and nice drawing of {{selected}}.",
  "This {{selected}} was drawn on a Nokia 3310.",
  "A friendly drawing of {{selected}}.",
  "The idea of {{selected}} comes through nicely here.",
  "This shows {{selected}} in a softer style.",
  "A calm and thoughtful drawing of {{selected}}.",
  "This is a recognizable interpretation of {{selected}}.",
  "My notes say this is {{selected}}.",
];

const dailyChallengeMessages = [
  "Well done, that’s today’s {{selected}} challenge done.",
  "You wrapped up the daily challenge with {{selected}}.",
  "That’s a solid {{selected}}, daily challenge finished.",
  "Good job, {{selected}} closes out today’s challenge.",
  "Today’s challenge done, you've unlocked the golden guess button!",
  "That’s it for today! New challenge tomorrow.",
  "Challenge complete! Check back tomorrow for another country.",
  "You’re done for today, tomorrow brings a fresh challenge.",
  "Daily challenge cleared. See you tomorrow!",
];

const countryFacts = {
  Afghanistan:
    "Afghanistan's landscape is dominated by the Hindu Kush mountain range.",
  Albania:
    "Albania has more bunkers per square kilometer than any other country.",
  Algeria: "Algeria is the largest country in Africa by land area.",
  Angola: "Angola's Kalandula Falls are among Africa's largest waterfalls.",
  "Antigua & Barbuda":
    "Antigua and Barbuda boasts 365 beaches, one for each day of the year.",
  Argentina:
    "Argentina is home to the highest peak in the Americas, Aconcagua.",
  Armenia:
    "Armenia's Lake Sevan is one of the largest high-altitude lakes in the world.",
  Australia: "Australia is the only country that is also a continent.",
  Austria: "Austria's Alps cover over 60% of the country.",
  Azerbaijan:
    "Azerbaijan is home to the world's first oil well, drilled in 1846.",
  Bahamas: "The Bahamas comprises around 700 islands and over 2,000 cays.",
  Bahrain:
    "Bahrain is an archipelago of 33 natural islands in the Persian Gulf.",
  Bangladesh:
    "Bangladesh contains the world's largest river delta, the Sundarbans.",
  Barbados: "Barbados is the easternmost island in the Caribbean.",
  Belarus: "Belarus has Europe's largest remaining primeval forest.",
  Belgium: "Belgium has three official languages: Dutch, French, and German.",
  Belize: "Belize hosts the second-largest barrier reef in the world.",
  Benin:
    "Benin has a narrow Atlantic coastline stretching just 121 kilometers.",
  Bhutan:
    "Bhutan measures its progress by Gross National Happiness instead of GDP.",
  Bolivia: "Bolivia is home to Salar de Uyuni, the world's largest salt flat.",
  "Bosnia & Herzegovina":
    "Bosnia and Herzegovina is home to the historic Stari Most bridge in Mostar, rebuilt after the war.",
  Botswana:
    "Botswana's Okavango Delta is one of the world's largest inland river deltas.",
  Brazil: "Brazil contains about 60% of the Amazon Rainforest.",
  Brunei:
    "Brunei hosts one of the world's largest water villages, Kampong Ayer.",
  Bulgaria:
    "Bulgaria is home to the oldest gold treasure in the world, found in Varna.",
  "Burkina Faso": "Burkina Faso's terrain is mostly flat savanna.",
  Burundi: "Burundi borders Lake Tanganyika, the world's second-deepest lake.",
  Cambodia:
    "Cambodia is home to Angkor Wat, the largest religious monument in the world.",
  Cameroon: "Cameroon's Mount Cameroon is the highest point in West Africa.",
  Canada: "Canada has more lakes than the rest of the world's lakes combined.",
  "Cape Verde":
    "Cape Verde is an archipelago of 10 volcanic islands in the central Atlantic Ocean.",
  "Central African Republic":
    "The Central African Republic lies entirely within the tropical zone.",
  Chad: "Chad's Lake Chad, once Africa's largest lake, has significantly shrunk over the decades.",
  Chile:
    "Chile contains the Atacama Desert, the driest non-polar desert on Earth.",
  China:
    "China shares the world's highest mountain, Mount Everest, with Nepal.",
  Colombia:
    "Colombia is the only South American country with coastlines on both the Pacific and Atlantic Oceans.",
  Comoros:
    "The Comoros islands were formed by volcanic activity in the Indian Ocean.",
  Congo: "The Congo River is Africa's second-longest and deepest river.",
  "Costa Rica": "Costa Rica contains 5% of the world's biodiversity.",
  Croatia: "Croatia boasts over 1,200 islands along its Adriatic coast.",
  Cuba: "Cuba is the largest island in the Caribbean.",
  Cyprus: "Cyprus is divided between Greek and Turkish territories.",
  "Czech Republic":
    "The Czech Republic's Prague Castle is the largest ancient castle in the world.",
  "Côte d'Ivoire":
    "The Ivory Coast is the world's largest producer of cocoa beans.",
  "DR Congo":
    "The Democratic Republic of the Congo contains half of Africa's rainforest.",
  Denmark: "Denmark includes over 400 named islands.",
  Djibouti:
    "Djibouti's Lake Assal is Africa's lowest point at 155m below sea level.",
  Dominica:
    "Dominica contains the world's second-largest hot spring, Boiling Lake.",
  "Dominican Republic":
    "The Dominican Republic shares the island of Hispaniola with Haiti.",
  Ecuador: "Ecuador has the equator running directly through its territory.",
  Egypt:
    "Egypt is home to the Great Pyramid of Giza, the only surviving ancient wonder.",
  "El Salvador":
    "El Salvador is known as the 'Land of Volcanoes' with over 20 active ones.",
  "Equatorial Guinea":
    "Equatorial Guinea is the only African country where Spanish is an official language.",
  Eritrea: "Eritrea has one of the longest coastlines along the Red Sea.",
  Estonia: "Estonia has over 50% of its territory covered by forest.",
  Ethiopia:
    "Ethiopia's Great Rift Valley splits the country from northeast to southwest.",
  "Falkland Islands": "The Falkland Islands have more sheep than people.",
  "Faroe Islands":
    "The Faroe Islands have no naturally growing trees due to strong winds.",
  Fiji: "Fiji comprises over 330 islands, of which about 110 are inhabited.",
  Finland:
    "Finland contains around 188,000 lakes, earning it the nickname 'Land of a Thousand Lakes'.",
  France: "France has the most time zones of any country, totaling 12.",
  "French Guiana":
    "French Guiana hosts the Guiana Space Centre, a major European spaceport.",
  Gabon: "Gabon has nearly 90% of its land covered by rainforest.",
  Gambia: "The Gambia is the smallest country on mainland Africa.",
  Georgia: "Georgia's Mount Shkhara reaches a height of 5,193 meters.",
  Germany:
    "Germany's Rhine River is one of Europe's longest and most important waterways.",
  Ghana: "Ghana's Lake Volta is one of the world's largest man-made lakes.",
  Greece: "Greece has more than 6,000 islands, though only 227 are inhabited.",
  Greenland: "Greenland is the world's largest island that isn't a continent.",
  Guadeloupe:
    "Guadeloupe is an overseas region of France located in the Caribbean.",
  Guam: "Guam is the westernmost territory of the United States.",
  Guatemala: "Guatemala has 37 volcanoes, with several still active.",
  Guinea: "Guinea's highlands are the source of the Niger River.",
  "Guinea-Bissau":
    "Guinea-Bissau features a complex system of islands and mangroves in the Bijagós Archipelago.",
  Guyana:
    "Guyana's Kaieteur Falls is one of the world's tallest single-drop waterfalls.",
  Haiti:
    "Haiti was the first post-colonial independent black-led nation in the world.",
  "Heard and McDonald Islands":
    "These islands are home to Australia's only active volcano, Big Ben.",
  Honduras: "Honduras is home to the ancient Mayan city of Copán.",
  Hungary:
    "Hungary's capital, Budapest, has more thermal springs than any other capital city.",
  Iceland: "Iceland has more than 130 active and inactive volcanoes.",
  India: "India is the only country with both lions and tigers in the wild.",
  Indonesia:
    "Indonesia is made up of over 17,000 islands, making it the world's largest archipelago.",
  Iran: "Iran has one of the world's oldest continuous major civilizations.",
  Iraq: "Iraq is home to the ancient city of Babylon, a UNESCO World Heritage site.",
  Ireland: "Ireland is known as the Emerald Isle for its lush green landscape.",
  "Isle of Man":
    "The Isle of Man is located in the Irish Sea and is not part of the UK or EU.",
  Israel: "Israel's Dead Sea is the lowest point on Earth's surface.",
  Italy: "Italy contains the world's smallest country, Vatican City.",
  Jamaica:
    "Jamaica's Blue Mountains are famous for their coffee and scenic views.",
  Japan: "Japan consists of over 6,800 islands.",
  Jordan:
    "Jordan's ancient city of Petra is carved entirely into rose-red sandstone cliffs.",
  Kazakhstan: "Kazakhstan is the largest landlocked country in the world.",
  Kenya:
    "Kenya's Great Rift Valley runs through the entire length of the country.",
  Kiribati: "Kiribati is the only country situated in all four hemispheres.",
  Kuwait: "Kuwait has no natural rivers or lakes in its territory.",
  Kyrgyzstan: "Kyrgyzstan has over 90% of its territory covered in mountains.",
  Laos: "Laos's western border is formed by the mighty Mekong River.",
  Latvia: "Latvia has over half of its territory covered by forest.",
  Lebanon:
    "In Lebanon, you can ski in the morning and swim in the Mediterranean Sea in the afternoon.",
  Lesotho:
    "Lesotho is the only country entirely above 1,000 meters in elevation.",
  Liberia: "Liberia is Africa's oldest republic, founded in 1847.",
  Libya: "Libya recorded the hottest temperature ever measured on Earth.",
  Lithuania:
    "Lithuania contains the geographic center of Europe near its capital, Vilnius.",
  Luxembourg:
    "Luxembourg is one of the smallest and wealthiest countries in the world.",
  Madagascar:
    "Madagascar is the world's fourth-largest island and 90% of its wildlife is found nowhere else.",
  Malawi:
    "Malawi's Lake Malawi contains more fish species than any other lake.",
  Malaysia:
    "Malaysia's Petronas Towers were once the world's tallest buildings.",
  Mali: "Mali's ancient city of Timbuktu was a center of learning in the 15th century.",
  Martinique:
    "Martinique's Mount Pelée erupted in 1902, destroying the city of Saint-Pierre.",
  Mauritania: "Mauritania's Richat Structure is visible from space.",
  Mauritius:
    "Mauritius was the only known habitat of the now-extinct dodo bird.",
  Mayotte:
    "40% of the population of Mayotte, an overseas department of France, is aged 15 or younger.",
  Mexico:
    "Mexico is home to the world's smallest volcano, Cuexcomate, standing at just 13 meters tall.",
  Moldova:
    "Moldova's landscape is characterized by gently rolling hills and numerous vineyards.",
  Mongolia: "Mongolia is the least densely populated country in the world.",
  Montenegro:
    "Montenegro is home to Europe's deepest canyon, the Tara River Canyon.",
  Morocco:
    "Morocco contains part of the world's largest hot desert, the Sahara.",
  Mozambique: "Mozambique has one of the longest coastlines in Africa.",
  Myanmar:
    "Myanmar's Irrawaddy River is crucial for transportation and agriculture.",
  Namibia:
    "Namibia contains the Namib Desert, one of the oldest deserts in the world.",
  Nepal:
    "Nepal contains eight of the world's ten highest peaks, including Mount Everest.",
  Netherlands: "The Netherlands has about a third of its land below sea level.",
  "Netherlands Antilles":
    "The Netherlands Antilles consisted of six Caribbean islands until 2010.",
  "New Caledonia":
    "New Caledonia is surrounded by the world's second-largest double barrier coral reef.",
  "New Zealand":
    "New Zealand has the world's longest place name: Taumatawhakatangihangakoauauotamateaturipukakapikimaungahoronukupokaiwhenuakitanatahu.",
  Nicaragua:
    "Nicaragua's Lake Nicaragua is the only freshwater lake containing sharks.",
  Niger: "Niger has over 80% of its territory covered by the Sahara Desert.",
  Nigeria: "Nigeria has the largest population in Africa.",
  "North Korea":
    "North Korea's Mount Paektu is the country's highest and most sacred peak.",
  "North Macedonia":
    "Macedonia's Lake Ohrid is one of Europe's deepest and oldest lakes.",
  Norway:
    "Norway features the world's longest road tunnel, the Lærdal Tunnel, at 24.5 km.",
  Oman: "Oman's northern landscape is dominated by the Al Hajar Mountains.",
  Pakistan: "Pakistan is home to K2, the second-highest mountain in the world.",
  Palau:
    "Palau features Jellyfish Lake, where millions of stingless jellyfish migrate daily.",
  Panama: "Panama's famous canal connects the Atlantic and Pacific Oceans.",
  "Papua New Guinea":
    "Papua New Guinea has over 850 indigenous languages, more than any other country.",
  Paraguay:
    "Paraguay is one of only two landlocked countries in South America.",
  Peru: "Peru's Machu Picchu is one of the New Seven Wonders of the World.",
  Philippines: "The Philippines consists of over 7,600 islands.",
  Poland:
    "Poland contains the Białowieża Forest, one of Europe's last primeval forests.",
  Portugal: "Portugal is the westernmost country of mainland Europe.",
  "Puerto Rico":
    "Puerto Rico contains El Yunque, the only tropical rainforest in the U.S. National Forest System.",
  Qatar: "Qatar has the highest GDP per capita in the world.",
  Reunion:
    "Reunion is home to one of the world's most active volcanoes, Piton de la Fournaise.",
  Romania:
    "Romania is home to the Transfăgărășan, one of the world's most scenic roads.",
  Russia: "Russia spans 11 time zones, more than any other country.",
  Rwanda:
    "Rwanda is known as the 'Land of a Thousand Hills' due to its mountainous terrain.",
  "Saint Lucia":
    "St. Lucia is home to the Pitons, two volcanic spires and a UNESCO World Heritage site.",
  Samoa:
    "Samoa is one of the first countries to see each new day due to its location.",
  "Sao Tome and Principe":
    "Sao Tome and Principe is Africa's second-smallest country by population.",
  "Saudi Arabia":
    "Saudi Arabia contains the world's largest sand desert, the Rub' al Khali.",
  Senegal: "Senegal is the westernmost country on the African mainland.",
  Serbia:
    "Serbia's capital, Belgrade, is built where the Danube River meets the Sava.",
  "Sierra Leone":
    "Sierra Leone is home to the Tacugama Chimpanzee Sanctuary, protecting endangered primates.",
  Slovakia: "Slovakia is home to over 6,000 caves, many open to the public.",
  Slovenia:
    "Slovenia has over 60% of its land covered by forest, making it one of Europe's greenest countries.",
  "Solomon Islands":
    "The Solomon Islands consist of nearly 1,000 islands in the South Pacific.",
  Somalia:
    "Somalia has the longest coastline in mainland Africa, stretching over 3,000 kilometers.",
  "South Africa":
    "South Africa uniquely has three capital cities: Pretoria, Bloemfontein, and Cape Town.",
  "South Georgia Islands":
    "South Georgia is home to vast penguin colonies and has no permanent residents.",
  "South Korea": "South Korea's Jeju Island is a UNESCO World Heritage site.",
  "South Sudan":
    "South Sudan is the youngest country in the world, gaining independence in 2011.",
  Spain:
    "Spain has the second-highest number of UNESCO World Heritage Sites in the world.",
  "Sri Lanka": "Sri Lanka is known as the 'Pearl of the Indian Ocean'.",
  Sudan: "Sudan has more pyramids than any other country, including Egypt.",
  Suriname: "Suriname has over 90% of its land covered by tropical rainforest.",
  Svalbard:
    "Svalbard hosts the Global Seed Vault, preserving seeds from around the world.",
  Eswatini:
    "Eswatini is one of the world’s last remaining absolute monarchies.",
  Sweden:
    "Sweden's Icehotel in Jukkasjärvi is rebuilt every year from ice and snow.",
  Switzerland:
    "Switzerland has over 7,000 lakes, with Lake Geneva being the largest.",
  Syria:
    "Syria's capital, Damascus, is considered one of the oldest continuously inhabited cities.",
  Taiwan:
    "Taiwan was home to the world's tallest building from 2004 to 2010, Taipei 101.",
  Tajikistan: "Tajikistan has over 90% of its territory covered by mountains.",
  Tanzania: "Tanzania's Mount Kilimanjaro is Africa's highest peak.",
  Thailand:
    "Thailand is the only Southeast Asian country never colonized by a European power.",
  "Timor-Leste":
    "Timor-Leste is one of the youngest countries in the world, gaining independence in 2002.",
  Togo: "Togo features a unique lagoon, Lake Togo, separated from the Atlantic by a narrow strip of land.",
  "Trinidad and Tobago":
    "Trinidad contains the world's largest natural deposit of asphalt at Pitch Lake.",
  Tunisia: "Tunisia's landscape is 40% Sahara Desert.",
  Turkey:
    "Turkey's largest city, Istanbul, spans two continents: Europe and Asia.",
  Turkmenistan:
    "Turkmenistan's Darvaza gas crater is known as the 'Door to Hell'.",
  "Turks and Caicos Islands":
    "The Turks and Caicos Islands are home to one of the largest coral reef systems in the world.",
  Uganda:
    "Uganda contains the source of the Nile River, the world's longest river.",
  Ukraine: "Ukraine is the largest country entirely within Europe.",
  "United Arab Emirates":
    "The UAE is home to the world's tallest building, the Burj Khalifa in Dubai.",
  "United Kingdom":
    "The UK's Loch Ness holds more freshwater than all lakes in England and Wales combined.",
  "United States Virgin Islands":
    "The Virgin Islands are one of the few U.S. territories where people drive on the left side of the road.",
  "United States of America":
    "The United States has the world's largest economy.",
  Uruguay:
    "Uruguay's capital, Montevideo, is the southernmost capital city in South America.",
  Uzbekistan:
    "Uzbekistan is one of only two doubly landlocked countries in the world.",
  Vanuatu: "Vanuatu is one of the most seismically active regions on Earth.",
  Venezuela:
    "Venezuela's Angel Falls is the world's highest uninterrupted waterfall.",
  Vietnam:
    "Vietnam's Ha Long Bay features thousands of limestone islands and islets.",
  "West Bank":
    "The West Bank contains many sites sacred to three major religions.",
  "Western Sahara":
    "Western Sahara has the world's largest phosphate reserves.",
  Yemen:
    "Yemen's Socotra Island hosts hundreds of plant species found nowhere else on Earth.",
  Zambia:
    "Zambia shares Victoria Falls, one of the world's largest waterfalls, with Zimbabwe.",
  Zimbabwe:
    "Zimbabwe shares Victoria Falls, one of the world's largest waterfalls.",
};

function showMessage(message) {
  document.getElementById("message").innerText = message;
}

export function setEmptyGuessMessage() {
  const message =
    Math.random() < 0.1
      ? easterEggMessages[Math.floor(Math.random() * easterEggMessages.length)]
      : "You first need to draw a country";

  showMessage(message);
}

function getRandomMessage(messageList, variables) {
  // Pick a random message from the list
  const randomIndex = Math.floor(Math.random() * messageList.length);
  let template = messageList[randomIndex];

  // Replace all placeholders dynamically
  for (const [key, value] of Object.entries(variables)) {
    const regex = new RegExp(`{{${key}}}`, "g");
    template = template.replace(regex, value);
  }

  return template;
}

export function setConfidenceBasedMessage(score, guessedCountry) {
  const messageList =
    score > 0.28
      ? highConfidenceMessages
      : score > 0.18
        ? mediumConfidenceMessages
        : lowConfidenceMessages;
  const message = getRandomMessage(messageList, { guessed: guessedCountry });
  showMessage(message);
}

function getCountryFact(country) {
  const funFact = countryFacts[country]
    ? `\n\nFun fact: ${countryFacts[country]}`
    : "";
  return funFact;
}

export function setCorrectGuessMessage(selectedCountry) {
  const messageList = checkDailyChallenge(selectedCountry)
    ? dailyChallengeMessages
    : correctMessages;

  const message =
    getRandomMessage(messageList, { selected: selectedCountry }) +
    getCountryFact(selectedCountry);

  showMessage(message);
}

export function setIncorrectGuessMessage(selectedCountry, guessedCountry) {
  if (selectedCountry == "Other") {
    showMessage("I thought I knew all the countries... I guess not!");
    return;
  }
  const message = getRandomMessage(incorrectMessages, {
    selected: selectedCountry,
    guessed: guessedCountry,
  });

  showMessage(message);
}

let leaderboardMessageCache = {};

export function clearLeaderboardMessageCache() {
  leaderboardMessageCache = {};
}

export function setLeaderboardMessage(rank, total, props) {
  // Check cache for message
  if (leaderboardMessageCache[rank]) {
    showMessage(leaderboardMessageCache[rank]);
    return;
  }

  // Add medals for top 3 ranks
  const medals = ["🥇", "🥈", "🥉"];
  const medal = medals[rank] || "";

  const scorePercent = Math.round(props.country_score * 100);
  let message = `#${rank + 1} / ${total}\u00A0\u00A0\u00A0 | \u00A0\u00A0\u00A0Score: ${scorePercent}%\n${medal} `;

  const messageList =
    props.country_score > 0.5
      ? leaderboardMessagesHigh
      : leaderboardMessagesLow;

  message += getRandomMessage(messageList, {
    selected: props.country_name,
    guessed: props.country_guess,
  });

  // Cache the message
  leaderboardMessageCache[rank] = message;

  showMessage(message);
}
