#pragma once

#include <string>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/tokenizers/owTurkishStemmer.hpp"
#include "OpenWhiz/text/tokenizers/owFrenchStemmer.hpp"

namespace ow {

// Language-dispatching stemmer facade: routes stem() to the right per-language
// implementation. English (no inflection worth stripping) is a no-op passthrough.
class owStemmer {
public:
    explicit owStemmer(owLanguage language = owLanguage::English) : m_language(language) {}

    std::string stem(const std::string& word) const {
        switch (m_language) {
            case owLanguage::Turkish:
                return m_turkish.stem(word);
            case owLanguage::French:
                return m_french.stem(word);
            default:
                return word; // English: no-op passthrough, no suffix stripping needed.
        }
    }

private:
    owLanguage m_language;
    owTurkishStemmer m_turkish;
    owFrenchStemmer m_french;
};

} // namespace ow
