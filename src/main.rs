use std::{env, fs, io};

fn main() {
    let mut args = env::args();

    match args.nth(1) {
        Some(path) => match fs::read_to_string(path) {
            Ok(source) => run(source),
            Err(err) => panic!("Error reading file: {err}"),
        },
        None => {
            // run repl
            eprint!("> ");
            while let Some(Ok(line)) = io::stdin().lines().next() {
                run(line);
                eprint!("\n> ");
            }
            eprintln!("Goodbye! o/");
        }
    }
}

fn run(source: String) {
    let tokens = scanner::scan(source);
    dbg!(&tokens);

    let ast = parser::Parser::new(tokens).parse();
    dbg!(&ast);
}

mod scanner {
    pub fn scan(source: String) -> Vec<Token> {
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
                            break 'string Error(ScanError::UnterminatedStringLiteral);
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
                    while let Some((_, c)) =
                        source.next_if(|(_, c)| c.is_alphanumeric() || *c == '_')
                    {
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
                char => Error(ScanError::UnexpectedChar(char)),
            };

            let token = Token { data, pos };

            tokens.push(token);
        }

        tokens
    }

    #[derive(Clone, Debug)]
    pub struct Token {
        pub data: TokenType,
        pub pos: usize,
    }

    #[derive(Clone, Debug)]
    pub enum TokenType {
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

        Error(ScanError),
    }

    #[derive(Copy, Clone, Debug)]
    pub enum ScanError {
        UnexpectedChar(char),
        UnterminatedStringLiteral,
    }
}

mod parser {
    use std::iter::Peekable;

    use crate::{
        ast::{
            Binary, Comparison, ComparisonOperator, EqualityOperator, Expr, Factor, FactorOperator,
            Primary, Term, TermOperator, Unary, UnaryOperator,
        },
        scanner::{Token, TokenType},
    };

    pub struct Parser<T> {
        tokens: T,
    }

    impl<T: Iterator<Item = Token>> Parser<Peekable<T>> {
        pub fn new<I: IntoIterator<IntoIter = T>>(tokens: I) -> Self {
            let tokens = tokens.into_iter().peekable();
            Parser { tokens }
        }

        pub fn parse(&mut self) -> Expr {
            self.expression()
        }

        fn binary<Operand, Operator>(
            &mut self,
            mut operand: impl FnMut(&mut Self) -> Operand,
            operator: impl Fn(&TokenType) -> Option<Operator>,
        ) -> Binary<Operand, Operator> {
            let lhs = operand(self);
            let rhs = self
                .tokens
                .peek()
                .and_then(|t| operator(&t.data))
                .map(|op| {
                    // advance iterator if `operator()` matched
                    self.tokens.next();
                    op
                })
                .map(|op| (op, operand(self)));

            Binary { lhs, rhs }
        }

        fn expression(&mut self) -> Expr {
            Expr::Equality(self.equality())
        }

        fn equality(&mut self) -> Binary<Comparison, EqualityOperator> {
            self.binary(Self::comparison, |token| match token {
                TokenType::BangEqual => Some(EqualityOperator::NotEqual),
                TokenType::EqualEqual => Some(EqualityOperator::Equal),
                _ => None,
            })
        }

        fn comparison(&mut self) -> Comparison {
            self.binary(Self::term, |token| match token {
                TokenType::Greater => Some(ComparisonOperator::Greater),
                TokenType::GreaterEqual => Some(ComparisonOperator::GreaterEqual),
                TokenType::Less => Some(ComparisonOperator::Less),
                TokenType::LessEqual => Some(ComparisonOperator::LessEqual),
                _ => None,
            })
        }

        fn term(&mut self) -> Term {
            self.binary(Self::factor, |token| match token {
                TokenType::Minus => Some(TermOperator::Subtract),
                TokenType::Plus => Some(TermOperator::Add),
                _ => None,
            })
        }

        fn factor(&mut self) -> Factor {
            self.binary(Self::unary, |token| match token {
                TokenType::Slash => Some(FactorOperator::Divide),
                TokenType::Star => Some(FactorOperator::Multiply),
                _ => None,
            })
        }

        fn unary(&mut self) -> Unary {
            let operator = self
                .tokens
                .peek()
                .and_then(|token| match &token.data {
                    TokenType::Bang => Some(UnaryOperator::Not),
                    TokenType::Minus => Some(UnaryOperator::Negate),
                    _ => None,
                })
                .map(|op| {
                    self.tokens.next();
                    op
                });

            if let Some(operator) = operator {
                let unary = Box::new(self.unary());
                Unary::Unary { operator, unary }
            } else {
                Unary::Primary(self.primary())
            }
        }

        fn primary(&mut self) -> Primary {
            let Some(next_token) = self.tokens.next() else {
                todo!()
            };

            use Primary::{False, Nil, Number, String, True};
            match &next_token.data {
                TokenType::False => False,
                TokenType::True => True,
                TokenType::Nil => Nil,

                TokenType::Number(n) => Number(*n),
                TokenType::String(s) => String(s.clone()),

                TokenType::LeftParen => {
                    let expr = self.expression();
                    let Some(TokenType::RightParen) = self.tokens.peek().map(|t| &t.data) else {
                        panic!("Expected right paren");
                    };
                    Primary::Grouping(Box::new(expr))
                }

                token => panic!("Expected primary, got {token:#?}"),
            }
        }
    }
}

mod ast {
    #[derive(Debug)]
    pub struct Binary<Operand, Operator> {
        pub lhs: Operand,
        pub rhs: Option<(Operator, Operand)>,
    }

    #[derive(Debug)]
    pub enum Expr {
        Equality(Binary<Comparison, EqualityOperator>),
    }
    #[derive(Debug)]
    pub enum EqualityOperator {
        Equal,
        NotEqual,
    }

    pub type Comparison = Binary<Term, ComparisonOperator>;
    #[derive(Debug)]
    pub enum ComparisonOperator {
        Greater,
        GreaterEqual,
        Less,
        LessEqual,
    }

    pub type Term = Binary<Factor, TermOperator>;
    #[derive(Debug)]
    pub enum TermOperator {
        Subtract,
        Add,
    }

    pub type Factor = Binary<Unary, FactorOperator>;
    #[derive(Debug)]
    pub enum FactorOperator {
        Divide,
        Multiply,
    }

    #[derive(Debug)]
    pub enum Unary {
        Unary {
            operator: UnaryOperator,
            unary: Box<Unary>,
        },
        Primary(Primary),
    }
    #[derive(Debug)]
    pub enum UnaryOperator {
        Not,
        Negate,
    }

    #[derive(Debug)]
    pub enum Primary {
        Number(f64),
        String(String),
        True,
        False,
        Nil,
        Grouping(Box<Expr>),
    }
}
