// No-op stand-in for the theme-writer used at build time; all deck colours are explicit hex,
// so decks render identically without it (only PowerPoint's palette name differs).
module.exports = { applyTheme: async () => {} };
