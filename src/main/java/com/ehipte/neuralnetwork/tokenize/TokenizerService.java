package com.ehipte.neuralnetwork.tokenize;

import com.ehipte.neuralnetwork.tokenize.model.Pair;
import com.ehipte.neuralnetwork.tokenize.model.PairToUpdate;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Класс осуществляющий токенизацию входящего текста.
 */
public class TokenizerService {

    private static final Pattern PATTERN = Pattern.compile("[\\p{L}\\p{N}]+|\\s+|[^\\p{L}\\p{N}\\s]");

    /**
     * Словарь токенов. Последовательность символов/символ/слово:идентификатор токена
     */
    private final Map<String, Integer> vocabulary = new HashMap<>();

    /**
     * Ранги слияний
     */
    private final Map<Integer, Pair> mergeRanks = new TreeMap<>();

    /**
     * Текст разбитый посимвольно.
     */
    private List<String> textAsSymbols = new ArrayList<>();

    /**
     * Слова в тексте в виде листа символов(строк):частота встречаемости в тексте
     */
    private Map<List<String>, Integer> uniqueWordsInSymbols = new HashMap<>();

    /**
     * Слова в тексте:частота встречаемости в тексте
     */
    private Map<String, Integer> wordToFrequency = new HashMap<>();


    /**
     * Тренировка токенизатора, инициализация и заполнение словаря до необходимых значений
     *
     * @param text           тренировочный текст
     * @param vocabularySize требуемый размер словаря
     */
    public void train(String text, int vocabularySize) {
        List<String> split = splitTextToWords(text);
        wordToFrequency = calculateFrequency(split);
        textAsSymbols = splitTextToSymbols(text);
        initializeVocabulary(textAsSymbols); // инициализация должна быть при вообще загрузке класса
        uniqueWordsInSymbols = splitWordsToSymbols(wordToFrequency);
        bpeProcessing(vocabularySize);
    }

    /**
     * Разбивает входящий текст на токены и кодирует их
     *
     * @param inputText входной текст
     * @return последовательность токенов
     */
    public List<Integer> encode(String inputText) {
        List<Integer> tokenizedText = new ArrayList<>();

        List<String> textToSymbols = splitTextToSymbols(inputText);

        Set<Pair> collectedPairs;

        boolean hasPairToUpdate = true;

        while (hasPairToUpdate) {
            collectedPairs = collectPairs(textToSymbols);
            hasPairToUpdate = updateTextWithPair(collectedPairs, textToSymbols);
        }

        for (String textToSymbol : textToSymbols) {
            tokenizedText.add(vocabulary.get(textToSymbol));
        }
        return tokenizedText;
    }

    /**
     * Преобразует последовательность токенов в текст
     *
     * @param tokens входящая последовательность токенов
     * @return преобразует токены в текст и возвращает строку
     */
    public String decode(List<Integer> tokens) {
        StringBuilder builder = new StringBuilder();
        for (Integer token : tokens) {
            for (Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
                if (entry.getValue().equals(token)) {
                    builder.append(entry.getKey());
                }
            }
        }
        return builder.toString();
    }

    /**
     * Обновляем текст согласно заданным парам. Обновляем пары согласно рангам слияний.
     *
     * @param collectedPairs пары из которых состоит текст
     * @param textToSymbols  текст для обновления
     * @return true если была произведена замена отдельных символов/последовательностей символов
     * на одну последовательность символов. Возвращает false больше нет известных возможных слияний.
     */
    private boolean updateTextWithPair(Set<Pair> collectedPairs, List<String> textToSymbols) {
        for (Map.Entry<Integer, Pair> pairEntry : mergeRanks.entrySet()) {
            Pair pair = pairEntry.getValue();
            if (collectedPairs.contains(pair)) {

                ListIterator<String> iter = textToSymbols.listIterator();
                while (iter.hasNext()) {
                    String current = iter.next();

                    if (current.equals(pair.firstSequence()) && iter.hasNext()) {
                        String next = iter.next();

                        if (next.equals(pair.secondSequence())) {
                            iter.remove();
                            iter.previous();
                            iter.set(pair.firstSequence() + pair.secondSequence());
                        } else {
                            iter.previous();
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }

    /**
     * Собирает все уникальные последовательности/пары символов в тексте
     *
     * @param textToSymbols текст разбитый изначально на символы, а в последствии преобразованный в последовательности/пары символов
     * @return возвращает множество пар
     */
    //получаем все уникальные пары символов/последовательностей символов
    private Set<Pair> collectPairs(List<String> textToSymbols) {
        Set<Pair> pairs = new HashSet<>();
        for (int i = 0; i < textToSymbols.size() - 1; i++) {
            Pair pair = new Pair(textToSymbols.get(i), textToSymbols.get(i + 1));
            pairs.add(pair);
        }
        return pairs;
    }

    /**
     * Разбиваем исходный текст на слова, знаки препинания и пробелы
     *
     * @param inputText входящий текст
     * @return лист слов, знаков препинания и пробелов
     */
    public List<String> splitTextToWords(String inputText) {
        List<String> split = new ArrayList<>();
        Matcher matcher = PATTERN.matcher(inputText);
        while (matcher.find()) {
            split.add(matcher.group());
        }
        return split;
    }

    /**
     * Разбиваем исходный текст на символы
     *
     * @param inputText входящий текст
     * @return возвращает весь текст как лист символов
     * (используются строки для того, чтобы впоследствии менять символы на пары и последовательности)
     */
    public List<String> splitTextToSymbols(String inputText) {
        return Arrays.stream(inputText.split("")).collect(Collectors.toList());
    }

    /**
     * Разбиваем пары слово:частота встречаемости на массив символов:частота встречаемости
     *
     * @param words словарь слов и их частота встречаемости в тексте
     * @return словарь слов в виде массива символов и их частота встречаемости в тексте
     * (используются строки для того, чтобы впоследствии менять символы на пары и последовательности)
     */
    public Map<List<String>, Integer> splitWordsToSymbols(Map<String, Integer> words) {
        Map<List<String>, Integer> wordsInSymbols = new HashMap<>();
        for (Map.Entry<String, Integer> entry : words.entrySet()) {
            char[] symbols = entry.getKey().toCharArray();
            List<String> symbolsInString = new ArrayList<>();
            for (char symbol : symbols) {
                symbolsInString.add(String.valueOf(symbol));
            }
            wordsInSymbols.put(symbolsInString, entry.getValue());
        }
        return wordsInSymbols;
    }

    /**
     * Подсчитываем встречаемость слова в тексте и формируем словарь слово:частота встречаемости в тексте
     *
     * @param text исходный текст разбитый на слова, знаки препинания и пробелы
     * @return словарь слов/знак препинания/пробел: частота встречаемости в тексте
     */
    public Map<String, Integer> calculateFrequency(List<String> text) {
        Set<String> words = new HashSet<>(text);
        Map<String, Integer> wordFrequencies = new HashMap<>();
        for (String word : words) {
            int frequency = 0;
            for (String string : text) {
                if (word.equals(string)) {
                    frequency++;
                }
            }
            wordFrequencies.putIfAbsent(word, frequency);
        }
        return wordFrequencies;
    }

    /**
     * Инициализируем словарь символами
     *
     * @param characters исходный текст разбитый посимвольно
     */
    // в будущем перевести всё на байты и инициализировать всеми возможными состояниями байта
    public void initializeVocabulary(List<String> characters) {
        //добавить спецсимвол <|endoftext|>
        for (String chars : characters) {
            vocabulary.putIfAbsent(chars, vocabulary.size());
        }
    }

    /**
     * Добавляем последовательность в словарь
     *
     * @param wordToPopulate добавляет последовательность символов в словарь
     */
    public void populateVocabulary(String wordToPopulate) {
        vocabulary.putIfAbsent(wordToPopulate, vocabulary.size());
    }

    /**
     * Вычисляем наиболее популярные пары и добавляем их в словарь и ранги слияний
     *
     * @param vocabularySize величина до которой будет производиться наполнение словаря
     */
    public void bpeProcessing(int vocabularySize) {
        while (vocabulary.size() < vocabularySize) {
            Map<PairToUpdate, Integer> pairsFrequency = new HashMap<>();

            for (Map.Entry<List<String>, Integer> entries : uniqueWordsInSymbols.entrySet()) {

                List<String> wordInSymbolSequence = entries.getKey();

                for (int i = 0; i < wordInSymbolSequence.size() - 1; i++) {

                    String sequence = wordInSymbolSequence.get(i) + wordInSymbolSequence.get(i + 1);
                    PairToUpdate pairToUpdate = new PairToUpdate(
                            sequence,
                            wordInSymbolSequence.get(i),
                            wordInSymbolSequence.get(i + 1)
                    );
                    if (pairsFrequency.containsKey(pairToUpdate)) {
                        Integer newFrequency = pairsFrequency.get(pairToUpdate) + entries.getValue();
                        pairsFrequency.put(pairToUpdate, newFrequency);
                    } else {
                        pairsFrequency.put(pairToUpdate, entries.getValue());
                    }
                }
            }

            Optional<Map.Entry<PairToUpdate, Integer>> opt = pairsFrequency.entrySet().stream().max(Map.Entry.comparingByValue());
            if (opt.isPresent()) {
                PairToUpdate pair = opt.get().getKey();
                populateVocabulary(pair.sequence());
                mergeRanks.put(mergeRanks.size(), new Pair(pair.firstSymbols(), pair.secondSymbols()));
                replaceSymbolsToCharSequenceInText(pair.sequence(), pair.firstSymbols(), pair.secondSymbols());
                replaceCharsToCharSequenceInMap(pair.sequence(), pair.firstSymbols(), pair.secondSymbols());
            }
        }
    }

    /**
     * Заменяем отдельные символы/последовательности символов на одну последовательность символов в тексте
     *
     * @param pair           новая последовательность
     * @param firstSequence  первый символ/последовательность
     * @param secondSequence второй символ/последовательность
     */
    public void replaceSymbolsToCharSequenceInText(String pair, String firstSequence, String secondSequence) {
        ListIterator<String> iter = textAsSymbols.listIterator();

        while (iter.hasNext()) {
            String current = iter.next();

            if (current.equals(firstSequence) && iter.hasNext()) {
                String next = iter.next();

                if (next.equals(secondSequence)) {
                    iter.remove();
                    iter.previous();
                    iter.set(pair);
                } else {
                    iter.previous();
                }
            }
        }
    }

    /**
     * Заменяем отдельные символы/последовательности символов на одну последовательность символов в uniqueWordsInSymbols
     *
     * @param pair           новая последовательность
     * @param firstSequence  первый символ/последовательность
     * @param secondSequence второй символ/последовательность
     */
    public void replaceCharsToCharSequenceInMap(String pair, String firstSequence, String secondSequence) {
        for (Map.Entry<List<String>, Integer> listIntegerEntry : uniqueWordsInSymbols.entrySet()) {
            List<String> currentWord = listIntegerEntry.getKey();
            ListIterator<String> iter = currentWord.listIterator();

            while (iter.hasNext()) {
                String current = iter.next();

                if (current.equals(firstSequence) && iter.hasNext()) {
                    String next = iter.next();

                    if (next.equals(secondSequence)) {
                        iter.remove();
                        iter.previous();
                        iter.set(pair);
                    } else {
                        iter.previous();
                    }
                }
            }
        }
    }

    public Map<String, Integer> getVocabulary() {
        return vocabulary;
    }
}
