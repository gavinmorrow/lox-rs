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
    todo!()
}

fn run(source: String) {
    let tokens = scan(source);

    for token in tokens {
        dbg!(token);
    }
}

#[derive(Clone, Debug)]
struct Token {
    data: TokenType,
    lexeme: String,
    line: u32,
}

#[derive(Copy, Clone, Debug)]
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
    Identifier,
    String,
    Number,

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
}
