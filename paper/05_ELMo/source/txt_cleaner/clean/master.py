import re


class MasterCleaner:
    def __init__(self, config=None, debug=False):
        """
        The class has many regexes and clean the text with "cleaning" method.
        :param config: dictionary that containing variables in "cleaner_config_default.json".
        you can edit the keywords in the file for change regex list.
        :param debug: option for seeing the change of text by every process. (default: False)
        DO NOT USE THIS OPTION WITH LOOP OPERATION!!!

        Below describes for type of regexes

        # unicode_break:
        Regex that searching broken character "�".
        Remove all sentence including that character.

        # not_contain_hangul:
        Regex that searching a sentence that has not two succeeding hangul characters.
        In that case, a sentence is whole foreign e.g. english.

        # contain_japanese:
        Regex that searching if there is any hiragana or katakana characters.
        This is very STRICT RULE, so can be changed later.

        # contain_chinese:
        Regex that searching if there is 5 succeeding hanja.
        But hanja frequently appeared in news articles, so this rule also CHANGEABLE.

        # full_english_sentence:
        Regex that searching 5 english phrases with dot at the end.
        If there is full english sentence in the article, this regex will remove all sentence.
        Generally, it will works, but since this is STRICT rule, it can change later.

        # line_breakers:
        Regex that substitute newline characters as a space, for almost str_content.
        The regex contain normal \n (Linux), and \r\n (Windows),
        and also if newline character is decoded as unicode backslash + n (\\n)

        # non_general_space:
        Regex that substitute non-general space characters, below is list of those spaces.
        chr(160)   \xa0    NO-BREAK SPACE (NBSP)
        chr(8192)   \u2000  EN QUAD
        chr(8193)   \u2001  EM QUAD
        chr(8194)   \u2002  EN SPACE
        chr(8195)   \u2003  EM SPACE
        chr(8196)   \u2004  THREE-PER-EM SPACE
        chr(8197)   \u2005  FOUR-PER-EM SPACE
        chr(8198)   \u2006  SIX-PE-EM SPACE
        chr(8199)   \u2007  FIGURE SPACE
        chr(8200)   \u2008  PUNCTUATION SPACE
        chr(8201)   \u2009  THIN SPACE
        chr(8202)   \u200A  HAIR SPACE
        chr(8203)   \u200B  ZERO WIDTH SPACE
        chr(8239)   \u202F  NARROW NO-BREAK SPACE
        chr(8287)   \u205F  MEDIUM MATHEMATICAL SPACE
        chr(12288)  \u3000  IDEOGRAPHIC SPACE
        chr(65279)  \uFEFF  ZERO WIDTH NO-BREAK SPACE

        as general space
        chr(32) \x20    SPACE

        # inside_bracket:
        Regex that searching the word or phrase which is inside in bracket, or parenthesis.
        There is six types of bracket

        (   chr(40) \x28    LEFT PARENTHESIS   //   )   chr(41) \x29    RIGHT PARENTHESIS
        <   chr(60) \x3C    LESS-THAN SIGN   //   > chr(62) \x3E    MORE-THAN SIGN
        [   chr(91) \x5B    LEFT SQUARE BRACKET   //   ]    chr(93) \x5D    RIGHT SQUARE BRACKET
        {   chr(123)    \x7B    LEFT CURLY BRACKET   //   } chr(125)    \x7D    RIGHT CURLY BRACKET
        【   chr(12304)  \u3010  LEFT BLACK LENTICULAR BRACKET // 】  chr(12305)  \u3011  RIGHT BLACK LENTICULAR BRACKET
        〔   chr(12308)  \u3014  LEFT TORTOISE SHELL BRACKET   //   〕   chr(12309)  \u3015  RIGHT TORTOISE SHELL BRACKET

        # non_absolute_characters:
        Regex that remove all characters not searched by regex.
        The regex search hangul, hanja, digits, (upper/lower) alphabet, dot, comma, question and exclamation mark.
        If you need more character set, you can edit this regex.

        # multiple_space:
        Search multiple spaces and shrink them to one.
        """
        self.debug = debug

        self.minimum_space_count = config['minimum_space_count']

        self.unicode_break = re.compile(chr(65533))
        self.not_contain_hangul = re.compile(r'^((?![가-힣]{2}).)+$')
        self.contain_japanese = re.compile(r'^.+?([ぁ-ゔ]+|[ァ-ヴー]+[々〆〤]).+?$')
        self.contain_chinese = re.compile(r' [一-龥]{5,}')
        self.full_english_sentence = re.compile(r'^.+?( [A-Za-z]+ [A-Za-z]+ [A-Za-z]+ [A-Za-z]+ [A-Za-z]+\.) .+$')

        self.line_breakers = re.compile(r'(\n|\r\n|\\n)+')
        self.non_general_space = re.compile(r'[\xa0\u2000-\u200B\u202F\u205F\u3000\uFEFF]')
        self.inside_bracket = re.compile(r'[\x28\x3C\x5B\x7B\u3010\u3014].+?[\x29\x3E\x5D\x7D\u3011\u3015]')
        self.non_absolute_characters = re.compile(r'[^가-힣A-Za-z.,?!0-9一-龥 ]')
        self.multiple_space = re.compile(r' {2,}')

    def print_debug_process(self, text, state):
        """
        Function for printing process by process
        :param text: input text
        :param state: information about process state
        :return: None
        """
        def _print_during_process(t, s):
            print('\n----- {} -----\n'.format(s))
            print(t)
            print('\n')
        if self.debug:
            _print_during_process(text, state)

    def cleaning(self, text):
        """
        The function for cleaning the text.
        :param text: input text (may concatenation of str_title and str_content)
        :return: cleaned text
        """
        def substitute_by_regex(t, reg, change_to, state):
            t = reg.sub(change_to, t)
            if self.debug:
                self.print_debug_process(t, state)
            return t

        def remove_by_regex(t, reg, state):
            return substitute_by_regex(t, reg, '', state)

        def remove_by_condition(t, condition, state):
            if condition:
                t = ''
            if self.debug:
                self.print_debug_process(t, state)
            return t

        text = remove_by_regex(text, self.unicode_break, 'after deleting broken text')
        text = remove_by_regex(text, self.not_contain_hangul, 'after deleting non-hangul article')
        text = remove_by_regex(text, self.contain_japanese, 'after deleting contain japanese sentence')
        text = remove_by_regex(text, self.contain_chinese, 'after deleting contain chinese sentence')
        text = remove_by_regex(text, self.full_english_sentence, 'after deleting contain full english sentence')

        text = substitute_by_regex(text, self.line_breakers, ' ', 'after substitute line_breakers')
        text = substitute_by_regex(text, self.non_general_space, ' ', 'after substitute non-general spaces')

        text = remove_by_regex(text, self.inside_bracket, 'after deleting phrases inside of brackets')

        text = substitute_by_regex(text, self.non_absolute_characters, ' ', 'after deleting useless special characters')
        text = substitute_by_regex(text, self.multiple_space, ' ', 'after shrinking multiple whitespaces as one')

        text = remove_by_condition(text, self.test_no_content(text), 'check there is no content at all')
        text = remove_by_condition(text, self.test_blank_sentence(text), 'check there is no sentence at all')
        text = remove_by_condition(text, self.test_length_cover(text), 'check sentence length is larger than constant')
        return text.strip()

    def test_length_cover(self, text):
        return text.count(' ') < self.minimum_space_count

    @staticmethod
    def test_blank_sentence(text):
        return text.strip() == ''

    @staticmethod
    def test_no_content(text):
        return text.strip(' ').endswith('\t')
