import unittest
import tinychain as tc


class User(tc.app.Model):
    def __init__(self):
        self.name = tc.app.Field(tc.String, max_len=250)
        self.email = tc.app.Field(tc.String, max_len=250, primary_key=True)
        self.password_hash = tc.app.Field(tc.Id, max_len=250)


class AuthService(tc.app.App):
    __uri__ = "/auth"

    @classmethod
    def models(self):
        return [tc.User]

    def create_user(self, user):
        return self.users.insert(user)

    def authenticate(self, email, password):
        user = self.users.get(email)
        return tc.If(
            user.password_hash != password.hash(),
            tc.error.NotAuthorized("wrong email or password"))


if __name__ == "__main__":
    unittest.main()
