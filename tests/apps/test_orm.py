import unittest
import testutils
import tinychain as tc


ERR_ALREADY_EXISTS = "{{email}} already has an account"


@tc.app.model
class User(tc.Instance):
    @staticmethod
    def schema():
        return [
            tc.Column("name", tc.String, 250),
            tc.Column("email", tc.EmailAddress, 250),
            tc.Column("password_hash", tc.Id, 250)
        ]


class AuthService(tc.app.App):
    __uri__ = tc.URI("/test/auth")

    @staticmethod
    def _export():
        return [User]

    # @tc.post_op
    # def create_user(self, name: tc.String, email: tc.EmailAddress, password: tc.String):
    #     # TODO: support automatic salt generation with PBKDF2
    #     return tc.If(self.graph.users.where(email=email).is_empty(),
    #                  self.graph.users.insert(User.load(name=name, email=email, password=password.hash())),
    #                  tc.error.BadRequest(ERR_ALREADY_EXISTS.render(email=email)))

    # @tc.post_op
    # def authenticate(self, email: tc.EmailAddress, password: tc.String):
    #     # TODO: support password validation with PBKDF2
    #     user = self.graph.users.get(email)
    #     return tc.If(
    #         user.password_hash != password.hash(),
    #         tc.error.NotAuthorized("wrong email or password"))


class ORMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_orm", [AuthService()])

    def testAuthService(self):
        user = {"name": "Test", "email": "test@example.com", "password": "12345"}
        self.assertRaises(tc.error.NotFound, lambda: self.host.post("/test/auth/create_user", user))


if __name__ == "__main__":
    unittest.main()
