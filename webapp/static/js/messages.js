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
  "Nice work — clearly {{selected}}!",
  "That’s a textbook {{selected}}!",
];

// Messages for incorrect predictions
export const incorrectMessages = [
  "Oh! It’s actually {{selected}} — I see why I thought it was {{guessed}}.",
  "Ah, {{selected}}! That does look a lot like {{guessed}}.",
  "Now I see — it’s {{selected}}, not {{guessed}}.",
  "Thanks for helping me learn the difference between {{selected}} and {{guessed}}!",
  "I confused {{selected}} with {{guessed}} — good to know for next time!",
  "{{selected}}! That one tripped me up.",
  "Looks like I mistook {{selected}} for {{guessed}}. I'm learning!",
  "I see it now — definitely {{selected}}, not {{guessed}}.",
  "Oops, that’s {{selected}}. I’ll remember the difference.",
  "{{selected}} — got it! I’ll be sharper next time.",
];

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

export function formatConfidence(score) {
  return `${(score * 100).toFixed(1)}%`;
}
