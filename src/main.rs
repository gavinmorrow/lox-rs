#![warn(clippy::all, clippy::pedantic)]

use std::{env, fs, io};

fn main() {
    let mut args = env::args();

    if let Some(path) = args.nth(1) {
        match fs::read_to_string(path) {
            Ok(source) => run(
                source,
                &mut interperter::env::Environment::new(),
                interperter::env::Scope::new(None),
            ),
            Err(err) => panic!("Error reading file: {err}"),
        }
    } else {
        // run repl
        let mut env = interperter::env::Environment::new();
        let scope = interperter::env::Scope::new(None);

        eprint!("> ");
        while let Some(Ok(line)) = io::stdin().lines().next() {
            run(line, &mut env, scope.clone());
            eprint!("> ");
        }
        eprintln!("Goodbye! o/");
    }
}

fn run(source: String, env: &mut interperter::env::Environment, scope: interperter::env::Scope) {
    let tokens = scanner::scan(source);
    let ast = parser::Parser::new(tokens).parse();
    match ast {
        Ok(ast) => match interperter::interpert(ast, env, scope) {
            Ok(()) => {}
            Err(err) => eprintln!("Runtime Error: {err:#?}"),
        },
        Err(err) => eprintln!("Error parsing tokens: {err:#?}"),
    }
}

mod scanner {
    pub fn scan(source: impl AsRef<str>) -> Vec<Token> {
        let mut tokens = vec![];

        let mut source = source.as_ref().char_indices().peekable();
        while let Some((pos, char)) = source.next() {
            use TokenType::{
                And, Bang, BangEqual, Class, Comma, Dot, Else, Equal, EqualEqual, Error, False,
                For, Fun, Greater, GreaterEqual, Identifier, If, LeftBrace, LeftParen, Less,
                LessEqual, Minus, Nil, Number, Or, Plus, Print, Return, RightBrace, RightParen,
                Semicolon, Slash, Star, String, Super, This, True, Var, While,
            };

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
        #[expect(
            dead_code,
            reason = "Haven't made great error messages yet, just uses Debug impl."
        )]
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

        #[expect(
            dead_code,
            reason = "Haven't made great error messages yet, just uses Debug impl."
        )]
        Error(ScanError),
    }

    #[expect(
        dead_code,
        reason = "Haven't made great error messages yet, just uses Debug impl."
    )]
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
            Assignment, Ast, Binary, Block, Comparison, ComparisonOperator, Declaration, Equality,
            EqualityOperator, Expr, Factor, FactorOperator, IfStmt, Primary, Stmt, Term,
            TermOperator, Unary, UnaryOperator, VarDecl,
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

        pub fn parse(&mut self) -> Result<Ast, ParseError> {
            let mut statements = vec![];

            while self.tokens.peek().is_some() {
                statements.push(self.declaration()?);
            }

            Ok(statements)
        }

        fn declaration(&mut self) -> Result<Declaration, ParseError> {
            if self.matches(|t| matches!(t.data, TokenType::Var)) {
                Ok(Declaration::VarDecl(self.var_decl()?))
            } else {
                Ok(Declaration::Statement(self.statement()?))
            }
        }

        fn var_decl(&mut self) -> Result<VarDecl, ParseError> {
            let token = self.consume(
                |t| matches!(&t.data, TokenType::Identifier(_)),
                ParseErrorType::ExpectedVarName,
            )?;
            let TokenType::Identifier(name) = token.data else {
                unreachable!();
            };

            let initializer = if self.matches(|t| matches!(&t.data, TokenType::Equal)) {
                Some(self.expression()?)
            } else {
                None
            };

            self.semicolon()?;

            Ok(VarDecl { name, initializer })
        }

        fn statement(&mut self) -> Result<Stmt, ParseError> {
            if self.matches(|t| matches!(t.data, TokenType::Print)) {
                let print_stmt = Stmt::Print(self.expression()?);
                self.semicolon()?;
                Ok(print_stmt)
            } else if self.matches(|t| matches!(t.data, TokenType::If)) {
                Ok(Stmt::If(self.if_stmt()?))
            } else if self.matches(|t| matches!(t.data, TokenType::LeftBrace)) {
                Ok(Stmt::Block(self.block()?))
            } else {
                let expr_stmt = Stmt::Expression(self.expression()?);
                self.semicolon()?;
                Ok(expr_stmt)
            }
        }

        fn binary<Operand, Operator>(
            &mut self,
            mut operand: impl FnMut(&mut Self) -> Result<Operand, ParseError>,
            operator: impl Fn(&TokenType) -> Option<Operator>,
        ) -> Result<Binary<Operand, Operator>, ParseError> {
            let lhs = operand(self)?;
            let mut expr = Binary { lhs, rhs: vec![] };

            // Allow for parsing a sequence (eg 1 + 2 + 3)
            while let Some(rhs) = self
                .tokens
                .peek()
                // add operator
                .and_then(|t| operator(&t.data))
                .inspect(|_| {
                    // advance iterator if `operator()` matched
                    self.tokens.next();
                })
                // add rhs operand
                .map(|op| operand(self).map(|operand| (op, operand)))
                .transpose()?
            {
                expr.rhs.push(rhs);
            }

            Ok(expr)
        }

        fn if_stmt(&mut self) -> Result<IfStmt, ParseError> {
            self.consume(
                |t| matches!(t.data, TokenType::LeftParen),
                ParseErrorType::ExpectedLeftParen,
            )?;
            let condition = self.expression()?;
            self.consume(
                |t| matches!(t.data, TokenType::RightParen),
                ParseErrorType::ExpectedRightParen,
            )?;

            let then_branch = Box::new(self.statement()?);
            let else_branch = if self.matches(|t| matches!(t.data, TokenType::Else)) {
                Some(Box::new(self.statement()?))
            } else {
                None
            };

            Ok(IfStmt {
                condition,
                then_branch,
                else_branch,
            })
        }

        fn block(&mut self) -> Result<Block, ParseError> {
            let mut stmts = vec![];
            loop {
                if self.matches(|t| matches!(t.data, TokenType::RightBrace)) {
                    break;
                } else if self.tokens.peek().is_none() {
                    return Err(ParseError {
                        error: ParseErrorType::ExpectedRightBrace,
                        location: None,
                    });
                }

                stmts.push(self.declaration()?);
            }
            Ok(stmts)
        }

        fn expression(&mut self) -> Result<Expr, ParseError> {
            Ok(Expr::Assignment(self.assignment()?))
        }

        fn assignment(&mut self) -> Result<Assignment, ParseError> {
            let target_token = self.tokens.peek().cloned();
            let expr = self.equality()?;

            if self.matches(|t| matches!(t.data, TokenType::Equal)) {
                let target = expr;
                let value = self.assignment()?;

                // Ensure target is a valid identifier
                if !(target.rhs.is_empty()
                    && target.lhs.rhs.is_empty()
                    && target.lhs.lhs.rhs.is_empty()
                    && target.lhs.lhs.lhs.rhs.is_empty())
                {
                    return Err(ParseError {
                        error: ParseErrorType::InvalidAssignmentTarget,
                        location: target_token,
                    });
                }
                let Unary::Primary(Primary::Identifier(name)) = target.lhs.lhs.lhs.lhs else {
                    return Err(ParseError {
                        error: ParseErrorType::InvalidAssignmentTarget,
                        location: target_token,
                    });
                };

                Ok(Assignment::Assignment {
                    name,
                    value: Box::new(value),
                })
            } else {
                Ok(Assignment::Equality(expr))
            }
        }

        fn equality(&mut self) -> Result<Equality, ParseError> {
            self.binary(Self::comparison, |token| match token {
                TokenType::BangEqual => Some(EqualityOperator::NotEqual),
                TokenType::EqualEqual => Some(EqualityOperator::Equal),
                _ => None,
            })
        }

        fn comparison(&mut self) -> Result<Comparison, ParseError> {
            self.binary(Self::term, |token| match token {
                TokenType::Greater => Some(ComparisonOperator::Greater),
                TokenType::GreaterEqual => Some(ComparisonOperator::GreaterEqual),
                TokenType::Less => Some(ComparisonOperator::Less),
                TokenType::LessEqual => Some(ComparisonOperator::LessEqual),
                _ => None,
            })
        }

        fn term(&mut self) -> Result<Term, ParseError> {
            self.binary(Self::factor, |token| match token {
                TokenType::Minus => Some(TermOperator::Subtract),
                TokenType::Plus => Some(TermOperator::Add),
                _ => None,
            })
        }

        fn factor(&mut self) -> Result<Factor, ParseError> {
            self.binary(Self::unary, |token| match token {
                TokenType::Slash => Some(FactorOperator::Divide),
                TokenType::Star => Some(FactorOperator::Multiply),
                _ => None,
            })
        }

        fn unary(&mut self) -> Result<Unary, ParseError> {
            let operator = self
                .tokens
                .peek()
                .and_then(|token| match &token.data {
                    TokenType::Bang => Some(UnaryOperator::Not),
                    TokenType::Minus => Some(UnaryOperator::Negate),
                    _ => None,
                })
                .inspect(|_| {
                    self.tokens.next();
                });

            if let Some(operator) = operator {
                let unary = Box::new(self.unary()?);
                Ok(Unary::Unary { operator, unary })
            } else {
                Ok(Unary::Primary(self.primary()?))
            }
        }

        fn primary(&mut self) -> Result<Primary, ParseError> {
            use Primary::{False, Identifier, Nil, Number, String, True};

            let Some(next_token) = self.tokens.next() else {
                return Err(ParseError::new(ParseErrorType::ExpectedPrimary, None));
            };

            match &next_token.data {
                TokenType::False => Ok(False),
                TokenType::True => Ok(True),
                TokenType::Nil => Ok(Nil),

                TokenType::Number(n) => Ok(Number(*n)),
                TokenType::String(s) => Ok(String(s.clone())),
                TokenType::Identifier(name) => Ok(Identifier(name.clone())),

                TokenType::LeftParen => {
                    let expr = self.expression()?;
                    self.consume(
                        |t| matches!(t.data, TokenType::RightParen),
                        ParseErrorType::ExpectedRightParen,
                    )?;
                    Ok(Primary::Grouping(Box::new(expr)))
                }

                _ => Err(ParseError::new(
                    ParseErrorType::ExpectedPrimary,
                    Some(&next_token),
                )),
            }
        }

        fn semicolon(&mut self) -> Result<Token, ParseError> {
            self.consume(
                |t| matches!(t.data, TokenType::Semicolon),
                ParseErrorType::ExpectedSemicolon,
            )
        }

        fn matches(&mut self, f: impl Fn(&Token) -> bool) -> bool {
            self.tokens.next_if(f).is_some()
        }

        fn consume(
            &mut self,
            f: impl Fn(&Token) -> bool,
            err_type: ParseErrorType,
        ) -> Result<Token, ParseError> {
            self.tokens
                .next_if(f)
                .ok_or(ParseError::new(err_type, self.tokens.peek()))
        }
    }

    #[expect(
        dead_code,
        reason = "Haven't made great error messages yet, just uses Debug impl."
    )]
    #[derive(Debug)]
    pub struct ParseError {
        error: ParseErrorType,
        location: Option<Token>,
    }
    impl ParseError {
        // Take `&Token` rather than just `Token` so that it's easier to use
        // `self.tokens.peek()` w/ it.
        pub fn new(error: ParseErrorType, location: Option<&Token>) -> Self {
            let location = location.cloned();
            ParseError { error, location }
        }
    }

    #[derive(Debug)]
    pub enum ParseErrorType {
        ExpectedPrimary,
        ExpectedRightParen,
        ExpectedSemicolon,
        ExpectedVarName,
        InvalidAssignmentTarget,
        ExpectedRightBrace,
        ExpectedLeftParen,
    }
}

mod interperter {
    use env::{Environment, Identifier, Scope};

    use crate::ast::{
        Assignment, Ast, Binary, ComparisonOperator, Declaration, EqualityOperator, Expr,
        FactorOperator, IfStmt, Primary, Stmt, TermOperator, Unary, UnaryOperator, VarDecl,
    };

    #[expect(
        clippy::needless_pass_by_value,
        reason = "Consistency with other that take a owned Scope."
    )]
    pub fn interpert(ast: Ast, env: &mut Environment, scope: Scope) -> Result<(), Error> {
        for declaration in ast {
            declaration.evaluate(env, scope.clone())?;
        }
        Ok(())
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Value {
        Nil,
        Boolean(bool),
        Number(f64),
        String(String),
    }

    impl Value {
        fn is_truthy(&self) -> bool {
            use Value::{Boolean, Nil};
            !matches!(self, Nil | Boolean(false))
        }
    }

    impl std::fmt::Display for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let str = match self {
                Value::Nil => "nil",
                Value::Boolean(b) => &b.to_string(),
                Value::Number(n) => &n.to_string(),
                Value::String(s) => s,
            };
            f.write_str(str)
        }
    }

    trait BinaryOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error>;
    }

    impl BinaryOperator for EqualityOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            match self {
                EqualityOperator::Equal => Ok(Value::Boolean(a == b)),
                EqualityOperator::NotEqual => Ok(Value::Boolean(a != b)),
            }
        }
    }

    impl BinaryOperator for ComparisonOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            use Value::{Boolean, Number};
            let (Number(a), Number(b)) = (a, b) else {
                return Err(Error::TypeError);
            };
            match self {
                ComparisonOperator::Less => Ok(Boolean(a < b)),
                ComparisonOperator::LessEqual => Ok(Boolean(a <= b)),
                ComparisonOperator::Greater => Ok(Boolean(a > b)),
                ComparisonOperator::GreaterEqual => Ok(Boolean(a >= b)),
            }
        }
    }

    impl BinaryOperator for TermOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            use Value::{Number, String};
            match self {
                TermOperator::Add => match (a, b) {
                    (Number(a), Number(b)) => Ok(Number(a + b)),
                    (String(a), String(b)) => Ok(String(a + b.as_str())),
                    _ => Err(Error::TypeError),
                },
                TermOperator::Subtract => {
                    let (Number(a), Number(b)) = (a, b) else {
                        return Err(Error::TypeError);
                    };
                    Ok(Number(a - b))
                }
            }
        }
    }

    impl BinaryOperator for FactorOperator {
        fn apply(&self, a: Value, b: Value) -> Result<Value, Error> {
            use Value::Number;
            let (Number(a), Number(b)) = (a, b) else {
                return Err(Error::TypeError);
            };

            match self {
                FactorOperator::Divide => Ok(Number(a / b)),
                FactorOperator::Multiply => Ok(Number(a * b)),
            }
        }
    }

    pub mod env {
        use std::{
            collections::HashMap,
            sync::atomic::{AtomicUsize, Ordering},
        };

        use super::Value;

        pub struct Environment {
            values: HashMap<Identifier, Value>,
        }

        impl Environment {
            pub fn new() -> Self {
                Environment {
                    values: HashMap::new(),
                }
            }

            pub fn define(&mut self, identifier: Identifier, value: Value) {
                self.values.insert(identifier, value);
            }

            fn resolve_scope(&self, identifier: &Identifier) -> Option<Identifier> {
                if self.values.contains_key(identifier) {
                    Some(identifier.clone())
                } else {
                    let mut id = identifier.clone();
                    while let Some(id) = id.in_parent_scope() {
                        if let Some(resolved_ident) = self.resolve_scope(&id) {
                            return Some(resolved_ident);
                        }
                    }
                    None
                }
            }

            pub fn get(&self, identifier: &Identifier) -> Option<&Value> {
                self.resolve_scope(identifier)
                    .and_then(|identifier| self.values.get(&identifier))
            }

            fn get_mut(&mut self, identifier: &Identifier) -> Option<&mut Value> {
                self.resolve_scope(identifier)
                    .and_then(|identifier| self.values.get_mut(&identifier))
            }

            /// Update a value if it exists.
            ///
            /// Returns `Ok(())` if the value exists and was updated, and `Err(())`
            /// otherwise.
            pub fn set(&mut self, identifier: &Identifier, value: Value) -> Result<(), ()> {
                if let Some(variable) = self.get_mut(identifier) {
                    *variable = value;
                    Ok(())
                } else {
                    Err(())
                }
            }
        }

        #[derive(Clone, Eq, PartialEq, Hash)]
        pub struct Identifier {
            scope: Scope,
            name: String,
        }
        impl Identifier {
            pub fn new(scope: Scope, name: String) -> Self {
                Self { scope, name }
            }

            pub fn in_parent_scope(&mut self) -> Option<Self> {
                self.scope.parent.clone().map(|parent_scope| {
                    *self = Identifier {
                        scope: *parent_scope,
                        name: self.name.clone(),
                    };
                    self.clone()
                })
            }
        }

        #[derive(Clone, Eq, PartialEq, Hash)]
        pub struct Scope {
            pub parent: Option<Box<Scope>>,
            id: ScopeId,
        }
        impl Scope {
            pub fn new(parent: Option<Box<Scope>>) -> Self {
                let id = NEXT_SCOPE_ID.fetch_add(
                    1,
                    // Use Ordering::SeqCst b/c I'm not confident that anything else would work properly, and this guarentees it.
                    Ordering::SeqCst,
                );
                Scope { parent, id }
            }

            pub fn nest(&self) -> Self {
                Scope::new(Some(Box::new(self.clone())))
            }
        }

        type ScopeId = usize;
        static NEXT_SCOPE_ID: AtomicUsize = AtomicUsize::new(0);
    }

    trait Evaluate {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error>;
    }

    impl<Operand: Evaluate, Operator: BinaryOperator> Evaluate for Binary<Operand, Operator> {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            let mut lhs: Value = self.lhs.evaluate(env, scope.clone())?;

            for (op, rhs) in self.rhs {
                let rhs: Value = rhs.evaluate(env, scope.clone())?;
                lhs = op.apply(lhs, rhs)?;
            }

            Ok(lhs)
        }
    }

    impl Evaluate for Declaration {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            match self {
                Declaration::VarDecl(var_decl) => var_decl.evaluate(env, scope),
                Declaration::Statement(stmt) => stmt.evaluate(env, scope),
            }
        }
    }

    impl Evaluate for VarDecl {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            let identifier = Identifier::new(scope.clone(), self.name);
            let value = self
                .initializer
                .map(|expr| expr.evaluate(env, scope))
                .transpose()?
                .unwrap_or(Value::Nil);

            env.define(identifier, value);
            Ok(Value::Nil)
        }
    }

    impl Evaluate for Stmt {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            match self {
                Stmt::Expression(expr) => {
                    expr.evaluate(env, scope)?;
                }
                Stmt::If(if_stmt) => {
                    if_stmt.evaluate(env, scope)?;
                }
                Stmt::Print(expr) => {
                    let value = expr.evaluate(env, scope)?;
                    println!("{value}");
                }
                Stmt::Block(stmts) => {
                    let scope = scope.nest();
                    for stmt in stmts {
                        stmt.evaluate(env, scope.clone())?;
                    }
                }
            };
            Ok(Value::Nil)
        }
    }

    impl Evaluate for Expr {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            match self {
                Expr::Assignment(assignment) => assignment.evaluate(env, scope),
            }
        }
    }

    impl Evaluate for IfStmt {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            let IfStmt {
                condition,
                then_branch,
                else_branch,
            } = self;
            let condition = condition.evaluate(env, scope.clone())?;
            if condition.is_truthy() {
                then_branch.evaluate(env, scope)?;
            } else if let Some(else_branch) = else_branch {
                else_branch.evaluate(env, scope)?;
            }

            Ok(Value::Nil)
        }
    }

    impl Evaluate for Assignment {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            match self {
                Assignment::Assignment { name, value } => {
                    let identifier = Identifier::new(scope.clone(), name);
                    let value = value.evaluate(env, scope)?;
                    env.set(&identifier, value.clone())
                        .map_err(|()| Error::VarNotDefinied)?;
                    Ok(value)
                }
                Assignment::Equality(equality) => equality.evaluate(env, scope),
            }
        }
    }

    impl Evaluate for Unary {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            match self {
                Unary::Unary { operator, unary } => match operator {
                    UnaryOperator::Not => {
                        let value: Value = (*unary).evaluate(env, scope)?;
                        let value = value.is_truthy();
                        Ok(Value::Boolean(!value))
                    }
                    UnaryOperator::Negate => {
                        let Value::Number(n) = (*unary).evaluate(env, scope)? else {
                            return Err(Error::TypeError);
                        };
                        Ok(Value::Number(-n))
                    }
                },
                Unary::Primary(primary) => primary.evaluate(env, scope),
            }
        }
    }

    impl Evaluate for Primary {
        fn evaluate(self, env: &mut Environment, scope: Scope) -> Result<Value, Error> {
            use Value::{Boolean, Nil, Number, String};
            match self {
                Primary::Number(n) => Ok(Number(n)),
                Primary::String(s) => Ok(String(s)),
                Primary::True => Ok(Boolean(true)),
                Primary::False => Ok(Boolean(false)),
                Primary::Nil => Ok(Nil),
                Primary::Grouping(expr) => expr.evaluate(env, scope),
                Primary::Identifier(name) => env
                    .get(&Identifier::new(scope, name))
                    .cloned()
                    .ok_or(Error::VarNotDefinied),
            }
        }
    }

    #[derive(Debug)]
    pub enum Error {
        TypeError,
        VarNotDefinied,
    }
}

mod ast {
    pub type Ast = Vec<Declaration>;

    #[derive(Debug)]
    pub enum Declaration {
        VarDecl(VarDecl),
        Statement(Stmt),
    }

    #[derive(Debug)]
    pub struct VarDecl {
        pub name: String,
        pub initializer: Option<Expr>,
    }

    #[derive(Debug)]
    pub enum Stmt {
        Expression(Expr),
        If(IfStmt),
        Print(Expr),
        Block(Block),
    }

    pub type Block = Vec<Declaration>;

    #[derive(Debug)]
    pub struct IfStmt {
        pub condition: Expr,
        pub then_branch: Box<Stmt>,
        pub else_branch: Option<Box<Stmt>>,
    }

    #[derive(Debug)]
    pub enum Expr {
        Assignment(Assignment),
    }

    #[derive(Debug)]
    pub enum Assignment {
        Assignment {
            name: String,
            value: Box<Assignment>,
        },
        Equality(Equality),
    }

    #[derive(Debug)]
    pub struct Binary<Operand, Operator> {
        pub lhs: Operand,
        pub rhs: Vec<(Operator, Operand)>,
    }

    pub type Equality = Binary<Comparison, EqualityOperator>;
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
        Identifier(String),
    }
}
