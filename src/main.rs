use std::{env, fs};

fn main() {
    let mut args = env::args();

    let Some(path) = args.nth(1) else {
        eprintln!("Usage: lox-rs [script]");
        return;
    };

    match fs::read_to_string(path) {
        Ok(source) => run(source),
        Err(err) => eprintln!("Error reading file: {err}"),
    }
}

fn scan(source: String) -> Vec<Token> {
    let mut tokens = vec![];

    let mut source = source.char_indices().peekable();
    while let Some((pos, char)) = source.next() {
        use TokenType::*;

        let mut equals_variant = |no_eq: TokenType, eq: TokenType| -> TokenType {
            match source.next_if(|(_, c)| *c == '=') {
                Some(_) => eq,
                None => no_eq,
            }
        };

        let data = match char {
            '(' => LeftParen,
            ')' => RightParen,
            '{' => LeftBrace,
            '}' => RightBrace,
            ',' => Comma,
            '.' => Dot,
            '-' => Minus,
            '+' => Plus,
            ';' => Semicolon,
            '*' => Star,

            '!' => equals_variant(Bang, BangEqual),
            '=' => equals_variant(Equal, EqualEqual),
            '<' => equals_variant(Less, LessEqual),
            '>' => equals_variant(Greater, GreaterEqual),

            '/' => match source.next_if(|(_, c)| *c == '/') {
                Some(_) => {
                    // consume until newline
                    while source.next_if(|(_, c)| *c != '\n').is_some() {}
                    continue;
                }
                None => Slash,
            },
            '"' => 'string: {
                let mut chars = vec![];
                loop {
                    let next = source.next();
                    if let Some((_, char)) = next {
                        if char == '"' {
                            break;
                        }
                        chars.push(char);
                    } else {
                        break 'string Error(ParseError::UnterminatedStringLiteral);
                    }
                }
                String(std::string::String::from_iter(chars))
            }
            digit if digit.is_ascii_digit() => {
                let mut chars = vec![digit];
                while let Some((_, digit)) = source.next_if(|(_, c)| c.is_ascii_digit()) {
                    chars.push(digit);
                }
                // todo: float literals
                // requires 2 chars of lookahead
                // if let Some((_, dot)) = source.next_if(|(_, c)| *c == '.') {
                //     chars.push(dot);
                //     while let Some((_, digit)) = source.next_if(|(_, c)| c.is_ascii_digit()) {
                //         chars.push(digit);
                //     }
                // }

                let num = std::string::String::from_iter(chars)
                    .parse::<f64>()
                    .expect("parsed number literal should be valid");
                Number(num)
            }

            whitespace if whitespace.is_whitespace() => continue,
            ident if ident.is_alphabetic() || ident == '_' => {
                let mut ident = vec![ident];
                while let Some((_, c)) = source.next_if(|(_, c)| c.is_alphanumeric() || *c == '_') {
                    ident.push(c);
                }
                let ident = std::string::String::from_iter(ident);
                match ident.as_str() {
                    "and" => And,
                    "class" => Class,
                    "else" => Else,
                    "false" => False,
                    "for" => For,
                    "fun" => Fun,
                    "if" => If,
                    "nil" => Nil,
                    "or" => Or,
                    "print" => Print,
                    "return" => Return,
                    "super" => Super,
                    "this" => This,
                    "true" => True,
                    "var" => Var,
                    "while" => While,
                    ident => Identifier(ident.to_string()),
                }
            }
            char => Error(ParseError::UnexpectedChar(char)),
        };

        let token = Token { data, pos };

        tokens.push(token);
    }

    tokens
}

fn run(source: String) {
    let tokens = scan(source);
    dbg!(tokens);
}

#[derive(Clone, Debug)]
struct Token {
    data: TokenType,
    pos: usize,
}

#[derive(Clone, Debug)]
enum TokenType {
    // Single-character tokens.
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,

    // One or two character tokens.
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    // Literals.
    Identifier(String),
    String(String),
    Number(f64),

    // Keywords.
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,

    Eof,

    Error(ParseError),
}

#[derive(Copy, Clone, Debug)]
enum ParseError {
    UnexpectedChar(char),
    UnterminatedStringLiteral,
}
