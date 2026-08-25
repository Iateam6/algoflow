from importlib import import_module

from django.urls import resolve


class VisaPackageContractMixin:
    module_name: str
    url_prefix: str

    def test_config_matches_prompt_registry(self):
        agent = import_module(f"{self.module_name}.agent")
        config = import_module(f"{self.module_name}.config")
        self.assertEqual(
            config.SUPPORTED_DOCUMENT_TYPES,
            frozenset(agent.build_prompt_registry()),
        )

    def test_adapter_is_visa_scoped(self):
        adapter_module = import_module(f"{self.module_name}.adapter")
        config = import_module(f"{self.module_name}.config")
        adapter = adapter_module.ADAPTER
        self.assertEqual(adapter.visa_type, config.VISA_TYPE)
        self.assertEqual(adapter.display_name, config.DISPLAY_NAME)
        self.assertEqual(adapter.cache_namespace, config.CACHE_NAMESPACE)
        rag_config = adapter.rag_config(tenant_id="tenant-a", case_id="case-a")
        self.assertEqual(rag_config.tenant_id, "tenant-a")
        self.assertEqual(rag_config.case_id, "case-a")
        self.assertIn(config.CACHE_NAMESPACE, rag_config.cache_root)

    def test_sync_and_async_routes_resolve(self):
        self.assertEqual(resolve(self.url_prefix).url_name, "index")
        create_match = resolve(f"{self.url_prefix}generate_doc/")
        download_match = resolve(
            f"{self.url_prefix}generate_doc/00000000-0000-0000-0000-000000000000/download/"
        )
        self.assertEqual(create_match.url_name, "create_generation")
        self.assertEqual(download_match.url_name, "download_generation")
        self.assertTrue(create_match.func.csrf_exempt)
        self.assertTrue(download_match.func.csrf_exempt)
